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

# Change gym to gymnasium
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from blocks import Cube, Trapezoid

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="compas.geometry.point")

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
    
def gaussian(loc, xlim, zlim, img_size=(64, 64), sigma_x=0.6, sigma_y=1.0):
    x, y = loc
    X, Y = np.meshgrid(np.linspace(*xlim, img_size[0]), np.linspace(zlim[1], zlim[0], img_size[1]))
    return np.exp(-(((X - x)**2 / (2 * sigma_x**2)) + ((Y - y)**2 / (2 * sigma_y**2)))).reshape(img_size)

class AssemblyEnv(CRA_Assembly):

    def __init__(self, task, max_blocks=5, xlim=(-5, 5), zlim=(0, 10), img_size=(64, 64), mu=0.8, density=1.0):
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
        self.current_step  = 0
        self.num_faces = 4
        
        self.reward_feature = self.get_reward_features(sigma_x=0.5, sigma_y=1.0)

        C, H, W = 2, *self.img_size
        self.state_feature = torch.zeros((C, H, W), dtype=torch.float32)
        # on affiche aussi les obstacles dès le départ
        self._render_obstacles()
        # on affiche aussi les goals
        self._render_goals()

    def reset(self, obstacles=None):
        self.delete_blocks()
        self.graph._max_node = -1
        # self.blocks = {}
        # self._add_support_block()
        self.obstacles = []
        # Reinitialize environment: add floor, reset targets and state feature
        self.add_block(Floor(xlim=self.xlim))
        self.num_targets_reached = 0
        C, H, W = 2, *self.img_size
        self.state_feature = torch.zeros((C, H, W), dtype=torch.float32)
        # et on ré‐affiche les obstacles
        self._render_obstacles()
        # et on ré‐affiche les goals
        self._render_goals()

    def _render_obstacles(self):
        """Overlay des obstacles sur les canaux 0 et 1."""
        # masque initial
        mask = torch.zeros(self.img_size, dtype=torch.bool)
        for obs in self.task.obstacles:
            obs_feat = render_block_2d(obs, xlim=self.xlim, zlim=self.zlim, img_size=self.img_size)
            mask |= obs_feat.to(torch.bool)
        # on met 1.0 là où il y a un obstacle sur les deux canaux
        self.state_feature[0][mask] = 1.5
        self.state_feature[1][mask] = 1.5

    def _render_goals(self):
        """Overlay des goals (targets) sur les canaux 0 et 1."""
        for target in self.task.targets:
            x = target[0]
            z = target[-1]
            col = round((x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.img_size[1] - 1))
            row = round((self.zlim[1] - z)    / (self.zlim[1] - self.zlim[0]) * (self.img_size[0] - 1))
            # on met 1.0 là où il y a un goal
            self.state_feature[0][row, col] = 2.0
            self.state_feature[1][row, col] = 2.0

    def get_reward_features(self, sigma_x=0, sigma_y=0):
        reward_features = np.zeros(self.img_size)
        if len(self.task.targets) == 0:
            return torch.zeros(self.img_size)

        for target in self.task.targets:
            x, y = target[0], target[-1]
            if sigma_x == 0 and sigma_y == 0:
                x = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
                y = (y - self.zlim[0]) / (self.zlim[1] - self.zlim[0])
                reward_features[round((1-y)*self.img_size[0]), round(x*self.img_size[1])] = 1
            else:
                reward_features += 1 * gaussian((x,y), self.xlim, self.zlim, self.img_size)
        # Normalize the reward features

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
        )
        
        mask = action_feature.to(torch.bool)      # shape (H,W)
        # which block index just got placed?
        # since you do self.block_list.append(new_block) in add_block(),
        # the index is len(self.block_list)-1
        block_idx = len(self.block_list) - 1
    
        # channel 0 gets (block_idx + 1)
        self.state_feature[0][mask] = (block_idx + 1) / float(self.max_blocks)
        self.state_feature[1][mask] = (action.face + 1) / float(self.num_faces)

        self.current_step  += 1

        reward = 0

        for target in self.task.targets:
            if new_block.contains_2d(target):
                self.num_targets_reached += 1
                if len(self.block_list) - 2 >= 0:
                    reward += 100
        
        reward = torch.sum(action_feature * self.reward_feature, dim=(-1, -2)).flatten()[0]
        terminated = (len(self.block_list)-1 >= self.max_blocks) | self.num_targets_reached == len(self.task.targets)
        
        return self.state_feature, reward, terminated

    def collision(self, new_block):
        return any(new_block.intersects_2d(b) for b in self.block_list + self.task.obstacles)
    
    def available_actions(self, floor_positions=None, num_block_offsets=1, overlap=0.2):
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
    
    def random_action(self, num_block_offsets=1, non_colliding=True, stable=True):
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

    def get_block(self, action):
        if action.target_block >= len(self.block_list):
            return False, 'none'
        block = self.block_list[action.target_block]
        if block.name.startswith("Cube"):
            return True, 'cube'
        elif action.target_face == 0 and (action.offset_x < -1.5 or action.offset_x > 1.5):
                return True, 'trapezoid_big'
        elif action.target_face == 3 and (action.offset_x < -0.5 or action.offset_x > 0.5):
                return True, 'trapezoid_small'
        elif action.offset_x < -1 or action.offset_x > 1:
                return True, 'trapezoid_medium'
        return False, 'invalid_block'

    def test(self, action : Action):
        # Check if the action is valid
        if action.target_block >= len(self.block_list):
            return None, 'invalid_target_block'
        new_block = self.create_block(action)
        if not self.collision(new_block):
            self.add_block(new_block)  
            if self.is_stable():
                self.delete_block(list(self.nodes())[-1]) # remove the added block
            else:
                self.delete_block(list(self.nodes())[-1])

                return None, 'unstable'
        else:
            return None, 'collision'
        if action.target_block == 0:
            return action, 'floor'
        return action, 'good'
    
    def get_target_reward(self, action: Action):

        # on crée le bloc uniquement pour récupérer son frame
        if action.target_block >= len(self.block_list):
            return 0
        new_block = self.create_block(action)
        action_feature = render_block_2d(
            new_block, 
            xlim=self.xlim, 
            zlim=self.zlim, 
            img_size=self.img_size
        )
        reward = torch.sum(action_feature * self.reward_feature, dim=(-1, -2)).flatten()[0]

        return reward

class AssemblyGymEnv(gym.Env):
    """
    Gym wrapper for the Assembly Environment
    """
    def __init__(self, task, max_blocks=5, xlim=(-5, 5), zlim=(0, 10), 
                 img_size=(64, 64), mu=0.8, density=1.0):
        super().__init__()
        # Store configuration locally for action_space bounds
        self.xlim = xlim
        self.max_blocks = max_blocks
        self.num_faces = 4
        
        # Initialize the original environment
        self.env = AssemblyEnv(task, max_blocks, xlim, zlim, img_size, mu, density)
        
        # Define valid shapes and continuous Box action space
        self.valid_shapes = [1, 5]
        n_offsets = 10
        self.n_offsets = n_offsets  # store offsets count on instance

        self.action_space = spaces.MultiDiscrete([
            self.env.max_blocks,     # target_block ∈ [0 .. max_blocks-1]
            4,                       # target_face  ∈ [0 .. 3]
            len(self.valid_shapes),  # shape_idx    ∈ [0 .. len(valid_shapes)-1]
            4,                       # face         ∈ [0 .. 3]
            n_offsets                # offset_idx   ∈ [0 .. n_offsets-1]
        ])
        # Define observation space (2D image representation)
        self.observation_space = spaces.Box(
        low=0.0,
        high=1.0,              # or max block/face ID you expect
        shape=(2, img_size[0], img_size[1]),
        dtype=np.float32
    )
        
        self.count = 0
        self.minus = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        self.env.reset()
        self.count = 0

        # already shape (2,H,W)
        obs = self.env.state_feature.numpy().astype(np.float32)
        return obs, {}
    
    def step(self, action_array):

        self.minus = len(self.env.block_list) - 2

        # action_array: [target_block, target_face, shape_idx, face, offset_x]
        if action_array is None:
            raise ValueError("Action cannot be None")

        tb, tf, si, fa, off_idx = action_array.astype(int)

        # 1) Forçage sol
        if tb == 0:
            tf = 0


        shape_id = self.valid_shapes[si]

        # 3) Conversion de l’offset discret en valeur réelle
        offset_values = np.linspace(-0.8, 0.8, self.n_offsets)
        ox = float(offset_values[off_idx])
        if tb == 0:
            ox *= 5.0  # même logique floor si tu veux

            
        action_obj = Action(target_block=tb, target_face=tf, shape=shape_id, face=fa, offset_x=ox)

        
        #check the block 
        valid, block_type = self.env.get_block(action_obj)
        if valid:
            if block_type == 'cube':
                action_obj.offset_x *= 1.0
            elif block_type == 'trapezoid_big':
                action_obj.offset_x *= 1.3
            elif block_type == 'trapezoid_medium':
                action_obj.offset_x *= 0.8
            elif block_type == 'trapezoid_small':
                action_obj.offset_x *= 0.5

        target_reward = self.env.get_target_reward(action_obj)
        
        action_obj, cause = self.env.test(action_obj)
        info = {
            'num_blocks': len(self.env.block_list)-1,
            'num_targets_reached': self.env.num_targets_reached,
            'cause': cause,
            'stack': 'none',
            'block': 'none',
            'target_reward_good': 0,
            'target_reward_bad': 0,
            'validate_reward_good': 0,
            'validate_reward_bad': 0,
        }
        # Invalid action
        negative_reward = -2
        if action_obj is None:
            obs = self.env.state_feature.numpy().astype(np.float32)
            if info['cause'] == 'collision':
                negative_reward = - 2
            elif info['cause'] == 'unstable':
                negative_reward = - 2
            elif info['cause'] == 'invalid_target_block':
                negative_reward = - 2
            if tb != 0:
                info['stack'] = 'try'
                negative_reward += 1
            else:
                if self.minus >= 0:
                    negative_reward -=  target_reward
                    info['target_reward_bad'] = target_reward
                else:
                    info['target_reward_good'] = target_reward
                info['stack'] = 'none'
            return obs, negative_reward, False, False, info
        # Execute valid action
        obs, reward, done = self.env.step(action_obj)
        # convert to numpy float32 to match observation_space
        obs = obs.numpy().astype(np.float32)
        if tb != 0: 
            info['stack'] = 'done'
            if self.count < 2:
                reward *= -1
                reward -= 5
        else:
            self.minus -= 1
            if self.minus >= 0:
                reward *= -2
                reward -= 5
                info['validate_reward_bad'] = reward
            else:
                if tb == 0:
                    self.count += 1
                reward -= 1
                reward *= 3
                info['validate_reward_good'] = reward
        return obs, reward, done, False, info
    

    
    def return_env(self):
        """Return the underlying environment"""
        return self.env
    
    def action_to_dict(self, action):
        return {
            'target_block': action.target_block,
            'target_face': action.target_face,
            'shape': action.shape,
            'face': action.face,
            'offset_x': float(action.offset_x)
        }
    
    def get_available_actions(self, floor_positions=None, num_block_offsets=1, overlap=0.2):
        actions = self.env.available_actions(floor_positions=floor_positions, num_block_offsets=num_block_offsets, overlap=overlap)
        return [self.action_to_dict(x) for x in actions]
    
    def random_action(self, num_block_offsets=1, non_colliding=False, stable=False):
        action = self.env.random_action(num_block_offsets=num_block_offsets, non_colliding=non_colliding, stable=stable)
        if action is None:
            return None
        return self.action_to_dict(action)
    
    def get_target_reward(self, action):
        return self.env.get_target_reward(action)

