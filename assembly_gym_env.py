from itertools import product
import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from tasks import Bridge
from assembly_env import AssemblyEnv, Action
from rendering import plot_assembly_env
from blocks import Floor


class AssemblyGymEnv(gym.Env):
    """Gym wrapper for the BlockAssembly environment"""
    def __init__(self, task, overlap=0.2, max_blocks=10):
        super().__init__()
        
        # Create the underlying environment
        self.task = task
        self.env = AssemblyEnv(task, max_blocks=max_blocks)
        self.overlap = overlap
        
        # Define action space parameters
        self.max_blocks = max_blocks
        self.faces_per_block = 4
        self.num_shapes = len(task.shapes) # Number of shapes
        self.num_offsets = len(task.floor_positions) 
        self.shape_mapping = {shape.block_id: shape for shape in task.shapes}

        # Calculate total action space size
        self.all_actions = list(
            product(
                range(self.max_blocks),
                range(self.faces_per_block),
                self.shape_mapping.keys(),
                range(self.faces_per_block),
                range(self.num_offsets),
            )
        )
        
        self.total_actions = len(self.all_actions)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.total_actions)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=self.env.img_size, 
            dtype=np.float32
        )
        
        # Initialize environment 
        self.available_actions = self._update_available_actions()
        self.current_action_mask = self._generate_action_mask(self.available_actions)
    
    def _update_available_actions(self):
        return self.env.available_actions(num_block_offsets=self.num_offsets)
    
    def _generate_action_mask(self, available_actions):
        action_mask = np.zeros(self.total_actions, dtype=np.float32)
        
        for action in available_actions:
            try:
                idx = self.action_to_idx(action)
                action_mask[idx] = 1.0
            except ValueError:
                # If action not found in mapping, skip it
                print(f"Warning: Action {action} not found in mapping, skipping.")
                continue
                
        return action_mask
  
    def idx_to_action(self, action_idx):
        if action_idx >= len(self.all_actions):
            raise ValueError(f"Action index {action_idx} out of range")
        
        target_block_idx, target_face, shape_idx, face, offset_idx = self.all_actions[action_idx]
        
        # Handle case where target block doesn't exist yet
        if target_block_idx >= len(self.env.block_list):
            raise ValueError(f"Target block index {target_block_idx} doesn't exist in environment")
            
        target_block = self.env.block_list[target_block_idx]
        shape = self.shape_mapping[shape_idx]
        
        # Calculate the actual offset
        if target_block.name == 'Floor':
            # For floor, use predefined positions
            if offset_idx < len(self.env.task.floor_positions):
                offset_x = self.env.task.floor_positions[offset_idx]
            else:
                offset_x = 0.0
        else:
            # For regular blocks, calculate the offset
            l1 = target_block.face_length_2d(target_face) 
            l2 = shape.face_length_2d(face)
                
            max_range = (1 - self.overlap) * (l1 + l2) / 2
            offsets = np.linspace(-max_range, max_range, self.num_offsets+2, endpoint=True)[1:-1]
            offset_x = offsets[offset_idx]
    
        return Action(target_block_idx, target_face, shape_idx, face, offset_x)
    
    
    def action_to_idx(self, action):
        target_block_idx = action.target_block
        target_face = action.target_face
        shape_idx = action.shape
        face = action.face
        offset_idx = 2 # Default offset index in the middle of the range
        
        # Calculate offset_frac_idx
        target_block = self.env.block_list[target_block_idx]
        shape = self.shape_mapping[shape_idx]
        
        if target_block.name == 'Floor':
            # For floor, use predefined positions
            offset_idx = np.where(np.isclose(self.env.task.floor_positions, action.offset_x))[0][0]
        else:
            # For regular blocks, calculate the offset index
            l1 = target_block.face_length_2d(target_face)
            l2 = shape.face_length_2d(face)
            max_range = (1 - self.overlap) * (l1 + l2) / 2
            offsets = np.linspace(-max_range, max_range, self.num_offsets+2, endpoint=True)[1:-1]
            offset_idx = np.where(np.isclose(offsets, action.offset_x))[0][0]
            
        # Create the action tuple and find its index in all_actions
        action_tuple = (target_block_idx, target_face, shape_idx, face, offset_idx)
        # Find the index of this action in the all_actions list
        try:
            action_idx = self.all_actions.index(action_tuple)
        except ValueError:
            # If not found, use the closest action
            closest_action_idx = 0
            min_distance = float('inf')
            for idx, act_tuple in enumerate(self.all_actions):
                # Calculate a distance metric between action_tuple and act_tuple
                distance = sum(abs(a - b) for a, b in zip(action_tuple, act_tuple))
                if distance < min_distance:
                    min_distance = distance
                    closest_action_idx = idx
            action_idx = closest_action_idx
            print(f"Warning: Action not found in all_actions, using closest action index: {action_idx}")
            
        return action_idx         
       
    def reset(self, seed=None, options=None):
        self.env.reset()
        self.env.add_block(Floor(xlim=self.env.xlim)) 
        self.env.num_targets_reached = 0
        self.env.state_feature = torch.zeros(self.env.img_size)
        
        # Update available actions and generate action mask
        self.available_actions = self._update_available_actions()
        self.current_action_mask = self._generate_action_mask(self.available_actions)
        
        # Return observation dictionary
        return self.env.state_feature.numpy(), {}
        
    def step(self, action_idx):
        # Check if the action is valid using the mask
        if self.current_action_mask[action_idx] == 0.0:
            valid_actions = np.where(self.current_action_mask == 1.0)[0]
            if len(valid_actions) > 0:
                # Choose a random valid action
                action_idx = np.random.choice(valid_actions)
            else:
                # No valid actions available, return current state with negative reward
                return self.env.state_feature.numpy(), 0.0, False, True, {'termination_reason': 'no_valid_actions'}
  
        # Convert action index to Action object
        try:
            action = self.idx_to_action(action_idx)
        except ValueError as e:
            print(f"Error converting action index to Action: {e}")
            return self.env.state_feature.numpy(), 0.0, False, False, {'termination_reason': 'no_mapping_idx_action'}

        state, reward, done = self.env.step(action)
        if state is None:
            return self.env.state_feature.numpy(), 0.0, done, False,{'termination_reason': 'failed_placement'}
        
        if not done:
            # Update available actions and action mask
            self.available_actions = self._update_available_actions()
            self.current_action_mask = self._generate_action_mask(self.available_actions)
        
        # Return the results
        return state.numpy(), reward.item(), done, False,{
            'targets_reached': f"{self.env.num_targets_reached}/{len(self.env.task.targets)}",
            'blocks_placed': len(self.env.block_list) - 1,  # Subtract 1 for the floor
        }
        
    def render(self, mode='human'):
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_assembly_env(self.env, fig=fig, ax=ax, task=self.env.task)
            plt.axis('equal')
            plt.show()
            return None
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented")
    
    def close(self):
        plt.close('all')
        
    def random_action(self, non_colliding=False, stable=False):
        available_actions = self._update_available_actions()
        
        if not available_actions:
            return None
        
        # Shuffle to randomize selection
        random.shuffle(available_actions)
        
        # If we don't need any filtering, just return the first action
        if not stable and not non_colliding and available_actions:
            action = available_actions[0]
            return self.action_to_idx(action)
        
            # Otherwise, filter actions based on criteria
        for action in available_actions:
            # Check for collision if requested
            if non_colliding:
                new_block = self.env.create_block(action)
                if self.env.collision(new_block):
                    continue
            
            # If we don't need stability check, return this action
            if not stable:
                return self.action_to_idx(action)
            
            # Check for stability if requested
            new_block = self.env.create_block(action)
            self.env.add_block(new_block)
            is_stable = self.env.is_stable()
            # Remove the block regardless of stability outcome
            self.env.delete_block(list(self.env.nodes())[-1])
            
            if is_stable:
                return self.action_to_idx(action)
        
        # No valid action found
        return None
    
# def main():
#     task = Bridge(num_stories=2)
#     wrapped_env = AssemblyGymEnv(task, max_blocks=2)
    
#     done = False
#     rewards = 0
#     while not done:
#         # Pick a random action
#         action_idx = wrapped_env.random_action(non_colliding=True, stable=True)
#         if action_idx is None:
#             break
#         obs, r, done, info = wrapped_env.step(action_idx)
#         print(info)

#         if done:
#             break

#         rewards += r
#     wrapped_env.render(mode='human')
    # obs, _ = wrapped_env.reset()
    
    
    # plt.figure(figsize=(8, 6))
    # plt.imshow(obs_img, cmap='viridis', origin='upper')
    # plt.colorbar(label='Feature Value')
    # plt.title('Initial State Feature Map')
    # plt.xlabel('X-axis')
    # plt.ylabel('Z-axis')
    # plt.show()
     
# if __name__ == "__main__":
#     main()