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
from tasks import Bridge, Tower
from assembly_env import AssemblyEnv, Action
from rendering import plot_assembly_env
from blocks import Floor

class AssemblyGymEnv(gym.Env):
    """Gym wrapper for the BlockAssembly environment"""
    def __init__(self, task, max_blocks=10, xlim=(-5, 5), zlim=(0, 10), 
                 img_size=(64, 64), mu=0.8, density=1.0, invalid_action_penalty=1.0,
                 failed_placement_penalty=0.5, truncated_penalty=1.0, max_steps=200,
                 state_representation='basic', reward_representation='basic'):
        super().__init__()
        
        self.max_steps = max_steps
        
        # Create the underlying environment
        self.task = task
        self.env = AssemblyEnv(
            task=task, 
            max_blocks=max_blocks,
            xlim=xlim,
            zlim=zlim,
            img_size=img_size,
            mu=mu,
            density=density,
            state_representation=state_representation,
            reward_representation=reward_representation,
        )
        
        # Penalty parameters
        self.failed_placement_penalty = failed_placement_penalty
        self.invalid_action_penalty = invalid_action_penalty
        self.truncated_penalty = truncated_penalty
        
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
            low=0, high=255, 
            shape=(1, *img_size), 
            dtype=np.float32,
        )
        
        self.reset()
    
    def _generate_action_mask(self,):
        
        available_actions = self.env.available_actions(num_block_offsets=self.num_offsets)
        self.action_mask = np.full(self.total_actions, False, dtype=bool)
        
        for action in available_actions:
            try:
                idx = self.action_to_idx(action)
                self.action_mask[idx] = True
            except ValueError:
                # If action not found in mapping, skip it
                print(f"Warning: Action {action} not found in mapping, skipping.")
                continue
    
    def get_action_masks(self):
        return self.action_mask
    
    def idx_to_action(self, action_idx, overlap=0.2):
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
                
            max_range = (1 - overlap) * (l1 + l2) / 2
            offsets = np.linspace(-max_range, max_range, self.num_offsets+2, endpoint=True)[1:-1]
            offset_x = offsets[offset_idx]
    
        return Action(target_block_idx, target_face, shape_idx, face, offset_x)
    
    
    def action_to_idx(self, action, overlap=0.2):
        target_block_idx = action.target_block
        target_face = action.target_face
        shape_idx = action.shape
        face = action.face
        offset_idx = 2 # Default offset index in the middle of the range
        
        target_block = self.env.block_list[target_block_idx]
        shape = self.shape_mapping[shape_idx]
        
        if target_block.name == 'Floor':
            # For floor, use predefined positions
            offset_idx = np.where(np.isclose(self.env.task.floor_positions, action.offset_x))[0][0]
        else:
            # For regular blocks, calculate the offset index
            l1 = target_block.face_length_2d(target_face)
            l2 = shape.face_length_2d(face)
            max_range = (1 - overlap) * (l1 + l2) / 2
            offsets = np.linspace(-max_range, max_range, self.num_offsets+2, endpoint=True)[1:-1]
            offset_idx = np.where(np.isclose(offsets, action.offset_x))[0][0]
            
        # Create the action tuple and find its index in all_actions
        action_tuple = (target_block_idx, target_face, shape_idx, face, offset_idx)
        try:
            action_idx = self.all_actions.index(action_tuple)
        except ValueError:
            print(f"Error: Action with idx {action_idx} not found in all_actions")
            
        return action_idx         
    
    def reset(self, seed=None, options=None):
        self.seed(seed)
        self.env.reset()
        obs = self.env.state_feature.numpy().reshape(1, *self.env.state_feature.shape)
        self.steps = 0
        self._generate_action_mask()
        
        # Return observation dictionary
        return obs, {}
    
    def _format_state(self):
        # Convert the state feature to a numpy array
        state = self.env.state_feature.numpy()
        
        # Reshape the state to match the observation space
        if len(state.shape) == 2:
            state = state.reshape(1, *state.shape)
        return state

    def step(self, action_idx):
        
        # Initialize info dict
        self.info = {
            'blocks_placed': 0,
            'targets_reached': None,
            'is_failed_placement': False,
            'is_invalid_action': False,
        }
        
        self.steps += 1
        step_reward = 0.0
        
        truncated = (self.steps >= self.max_steps)
        if truncated:
            step_reward -= self.truncated_penalty
        
        # Check for invalid action
        if self.action_mask[action_idx] == False:
            self.info['is_invalid_action'] = True
            step_reward -= self.invalid_action_penalty
            state = self._format_state()
            return state, step_reward, False, truncated, self.info
            
        action = self.idx_to_action(action_idx)
        
        # Execute the action
        state, reward, done = self.env.step(action)
        step_reward += reward.item()
        
        self.info['blocks_placed'] = len(self.env.block_list) - 1  # Subtract 1 for the floor
        self.info['targets_reached'] = f"{self.env.num_targets_reached}/{len(self.env.task.targets)}"
        # Handle failed placement
        if state is None:
            self.info['is_failed_placement'] = True
            step_reward -= self.failed_placement_penalty
            state = self._format_state()
            return state, step_reward, done, truncated, self.info
        if not done:
            # Update the action mask
            self._generate_action_mask()
            
        state = self._format_state()
        return state, step_reward, done, truncated, self.info

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        return [seed]
    
    def render(self, mode='human'):
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_assembly_env(self.env, fig=fig, ax=ax, task=self.env.task, equal=True, face_numbers=False)
            plt.show()
            
            return None
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented")
    
    def close(self):
        plt.close('all')
        
    
def main():
    
    # Create environment
    task = Tower(targets=[(0,3.5)], obstacles=[(0,0.5), (0, 1.5), (0, 2.5), (-1,0.5), (1,0.5), (-1,1.5), (-3.5, 0.5), (-3.5, 1.5), (4, 0.5), (4, 1.5)])
    wrapped_env = AssemblyGymEnv(
        task=task, 
        max_blocks=5, 
        state_representation='intensity', 
        reward_representation='reshaped'
    )

    done = False
    rewards = 0
    while not done:
        
        # Pick a random action
        action = wrapped_env.env.random_action(wrapped_env.num_offsets, non_colliding=True, stable=True)
        action_idx = wrapped_env.action_to_idx(action)
        if action_idx is None:
            print(f"Invalid action index {action_idx}, skipping...")
            break
        obs, r, done, truncated, info = wrapped_env.step(action_idx)
        print(f"Step: {wrapped_env.steps}, Action: {action}, Reward: {r}, Info: {info}")

        if done or truncated:
            break

        rewards += r
    
    wrapped_env.render(mode='human')
    
if __name__ == "__main__":
    main()