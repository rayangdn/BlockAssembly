import os
import numpy as np
import torch
import random
from dataclasses import dataclass

from compas.geometry import Frame
from compas_cra.datastructures import CRA_Assembly
from compas_cra.algorithms import assembly_interfaces_numpy

from blocks import Floor, block_from_id
from geometry import align_blocks

from rendering import render_block_2d

from stability import is_stable_rbe

import gym
from gym import spaces



@dataclass
class Action:
    target_block: int
    target_face: int
    shape: int
    face: int
    offset_x: float = 0.
    # offset_y: float = 0.
    # frozen: bool = False

    def __hash__(self):
        return hash((self.target_block, self.target_face, self.shape, self.face, self.offset_x))
    
def gaussian(loc, xlim, zlim, img_size=(64,64), sigma=2):
    x, y = loc
    X, Y = np.meshgrid(np.linspace(*xlim, img_size[0]), np.linspace(zlim[1], zlim[0], img_size[1]))
    return np.exp(-((X - x)**2 + (Y - y)**2) / (2*sigma**2)).reshape(img_size)

class AssemblyEnv(CRA_Assembly):

    def __init__(self, task, max_blocks=10, xlim=(-5, 5), zlim=(0, 10), img_size=(64, 64), mu=0.8, density=1.0):
        super().__init__()
        self.task = task
        self.xlim = xlim
        self.zlim = zlim
        self.img_size = img_size
        self.mu = mu
        self.density = density
        self.num_targets_reached = 0
        self.max_blocks = max_blocks
        # self.obstacles = []
        # self.blocks = {}
        #self.blocks = Blocks(self)
        self.block_list = [] # Blocks(self)
        self.add_block(Floor(xlim=self.xlim))
        
        self.reward_feature = self.get_reward_features(sigma=0.5)

        self.state_feature = torch.zeros(self.img_size)


    def reset(self, obstacles=None):
        self.delete_blocks()
        self.graph._max_node = -1
        # self.blocks = {}
        # self._add_support_block()
        self.obstacles = []

    def get_reward_features(self, sigma=1):
        reward_features = np.zeros(self.img_size)
        if len(self.task.targets) == 0:
            return torch.zeros(self.img_size)

        for target in self.task.targets:
            x, y = target[0], target[-1]
            if sigma == 0:
                x = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
                y = (y - self.zlim[0]) / (self.zlim[1] - self.zlim[0])
                reward_features[round((1-y)*self.img_size[0]), round(x*self.img_size[1])] = 1
            else:
                reward_features += gaussian((x,y), self.xlim, self.zlim, self.img_size, sigma)

        reward_features = torch.tensor(reward_features).float()
        
        # reward_features /= reward_features.sum()
        reward_features /= len(self.task.targets)
        # reward_features -= reward_features.max() / 2

        return reward_features
    
    def create_block(self, action : Action):
        block1 = self.block_list[action.target_block]
        block2 = block_from_id(action.shape)
        # coordinai = [action.offset_x, 0, action.offset_y]
        coordinates = [action.offset_x, 0, 0]

        new_block = align_blocks(block1=block1, face1=action.target_face, block2=block2, face2=action.face, frame1_coordinates=coordinates)
        return new_block
    
    def add_block(self, block, compute_interfaces=True):
        # if block.node is None:
        #     block.node = len(self.blocks)
        block.node = super().add_block(block, node=block.node)
        self.block_list.append(block)

        if block.fixed:
            self.set_boundary_condition(block.node)
            
        if compute_interfaces:
            self.compute_interfaces()

        return block.node

    def add_blocks(self, blocks):
        for block in blocks:
            self.add_block(block, compute_interfaces=False)
            
        node_index = {node: index for index, node in enumerate(self.nodes())}
            
        self.compute_interfaces()

    def step(self, action : Action):
        # create and add block to environment
        new_block = self.create_block(action)
        if self.collision(new_block):
            print("Collision")
            return None, torch.tensor(0.), True
        
        self.add_block(new_block)
        if not self.is_stable():
            print("Unstable")
            return None, torch.tensor(0.), True
        
        action_feature = render_block_2d(
            new_block, 
            xlim=self.xlim, 
            zlim=self.zlim, 
            img_size=self.img_size
        ).unsqueeze(0)
        
        self.state_feature = torch.minimum(
                self.state_feature + action_feature, 
                torch.tensor(1.0)
            )
        self.state_feature = self.add_targets_to_state(self.state_feature)
        
        for target in self.task.targets:
            if new_block.contains_2d(target):
                self.num_targets_reached += 1
        
        reward = torch.sum(action_feature * self.reward_feature, dim=(-1, -2)).flatten()[0]
        terminated = (len(self.block_list)-1 >= self.max_blocks) | self.num_targets_reached == len(self.task.targets)
        
        return self.state_feature, reward, terminated

    def collision(self, new_block):
        return any(new_block.intersects_2d(b) for b in self.block_list + self.task.obstacles)
    
    def available_actions(self, floor_positions=None, num_block_offsets=3, overlap=0.2):
        floor_positions = floor_positions or self.task.floor_positions
        actions = []
        
        for i, block in enumerate(self.block_list):
            for target_face in block.receiving_faces_2d():
                for shape in self.task.shapes:
                    for face in shape.attaching_faces_2d():
                        if block.name == 'Floor':
                            offsets = floor_positions
                        else:
                            # Generate offsets dynamically
                            l1 = block.face_length_2d(target_face)
                            l2 = shape.face_length_2d(face)
                            offset_range = (1 - overlap) * (l1 + l2) / 2
                            offsets = np.linspace(-offset_range, offset_range, num_block_offsets + 2, endpoint=True)[1:-1]
                            # print(f"Offsets: {offsets}")
                            
                        for offset_x in offsets:
                            actions.append(Action(i, target_face, shape.block_id, face, offset_x))
                            
        return actions
    
    def random_action(self, num_block_offsets=3, non_colliding=True, stable=True):
        # non_colliding = True requests an action that is not colliding
        # stable = True requests an action that is stable
        available_actions = self.available_actions(num_block_offsets=num_block_offsets)
        random.shuffle(available_actions)
        if not stable and (not non_colliding):
            return available_actions[0]
        for a in available_actions:
            new_block = self.create_block(a)
            if non_colliding and self.collision(new_block):
                continue
            if not stable:
                return a
            self.add_block(new_block)  
            if self.is_stable():
                self.delete_block(list(self.nodes())[-1]) # remove the added block
                return a
            self.delete_block(list(self.nodes())[-1])
        return None
    
    def set_boundary_condition(self, node):
        super().set_boundary_condition(node)
        self.compute_interfaces()

    def remove_boundary_condition(self, node):
        super().remove_boundary_condition(node)
        self.compute_interfaces()

    def delete_block(self, key):
        if not key in self.graph.node:
            raise ValueError(f"Block with key {key} not found.")
        
        block = self.graph.node[key]['block']
        del self._blocks[block.guid]
        del self.graph.node[key]
        super().delete_block(key)
        self.block_list.pop()
        # Todo: is this needed? we just need to delete the interface
        self.compute_interfaces()

    def delete_blocks(self, keys=None):
        if keys is None:
            keys = list(self.nodes())
        super().delete_blocks(keys)
        self.compute_interfaces()

    def is_stable(self):
        return is_stable_rbe(self)[0]

    def compute_interfaces(self):
        # delete all edges to avoid duplicates
        edges = [*self.graph.edges()]
        for edge in edges:
            self.graph.delete_edge(edge)
        if self.number_of_nodes() > 1:
            assembly_interfaces_numpy(self, amin=0.001)

    def add_targets_to_state(self, state):
        if state.dim() == 3 and state.shape[0] == 1:
            state = state.squeeze(0)
        # Add obstacles (red cubes, i.e., num_stories)
        if hasattr(self.task, 'obstacles') and self.task.obstacles is not None:
            for obs in self.task.obstacles:
                mask = render_block_2d(obs, xlim=self.xlim, zlim=self.zlim, img_size=self.img_size)
                state[mask.bool()] = 0.8  # Use 0.8 to represent a red block/obstacle
        # Add targets (green stars)
        for target in self.task.targets:
            # Create a fake block or a small mask at the target location
            # For now, just set the closest pixel
            x, z = target[0], target[-1]
            i = int((1 - (z - self.zlim[0]) / (self.zlim[1] - self.zlim[0])) * (self.img_size[0] - 1))
            j = int(((x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])) * (self.img_size[1] - 1))
            state[i, j] = 0.5  # Use 0.5 to represent a target
        return state

class GymAssemblyEnv(gym.Env):
    """Gym wrapper for AssemblyEnv"""
    
    def __init__(self, task, max_blocks=10, xlim=(-5, 5), zlim=(0, 10), img_size=(64, 64), mu=0.8, density=1.0):
        super().__init__()
        self.env = AssemblyEnv(task, max_blocks, xlim, zlim, img_size, mu, density)
        
        # Define observation space (2D image)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=img_size, 
            dtype=np.float32
        )
        
        # Define action space
        self.action_space = spaces.Discrete(1000)  # Placeholder, will be updated
        
        # Store action mapping
        self._action_mapping = None
        self._update_action_mapping()
        
        # For seeding
        self._np_random = None
    
    def _update_action_mapping(self):
        """Update the mapping between discrete actions and actual assembly actions"""
        available_actions = self.env.available_actions()
        self._action_mapping = available_actions
        # Set minimum size of 1 for the action space to avoid assertion error
        n_actions = max(1, len(available_actions))
        self.action_space = spaces.Discrete(n_actions)
    
    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation"""
        super().reset(seed=seed)
        
        # Set the seed for numpy random number generator
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        
        self.env.reset()
        # Add the floor block
        self.env.add_block(Floor(xlim=self.env.xlim))
        
        self._update_action_mapping()
        return self.env.state_feature.numpy(), {}  # Return observation and info dict
    
    def step(self, action):
        """Execute one step in the environment"""
        # If there are no available actions, return with done=True
        if len(self._action_mapping) == 0:
            return self.env.state_feature.numpy(), 0, True, False, {}
            
        # Convert discrete action to actual assembly action
        if action >= len(self._action_mapping):
            return self.env.state_feature.numpy(), 0, True, False, {}
            
        actual_action = self._action_mapping[action]
        state, reward, done = self.env.step(actual_action)
        
        # If state is None (collision or instability), return current state
        if state is None:
            negative_reward = -10
            return self.env.state_feature.numpy(), negative_reward, True, False, {"failure": True}
        
        # Update action mapping after step
        self._update_action_mapping()
        
        return state.numpy(), reward, done, False, {}  # Return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the environment"""
        # You can implement rendering here if needed
        pass
    
    def close(self):
        """Clean up resources"""
        pass