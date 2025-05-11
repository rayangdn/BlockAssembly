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
    
def gaussian(loc, xlim, zlim, sigma_x, sigma_y, img_size=(64, 64)):
    x, y = loc
    X, Y = np.meshgrid(np.linspace(*xlim, img_size[0]), np.linspace(zlim[1], zlim[0], img_size[1]))
    return np.exp(-(((X - x)**2 / (2 * sigma_x**2)) + ((Y - y)**2 / (2 * sigma_y**2)))).reshape(img_size)

class AssemblyEnv(CRA_Assembly):

    def __init__(self, task, min_block_reach_target, target_reward_per_block, max_blocks=6, xlim=(-5, 5), zlim=(0, 10), img_size=(64, 64), mu=0.8, density=1.0, not_reached_penalty=25):
        super().__init__()
        self.task = task
        self.xlim = xlim
        self.zlim = zlim
        self.img_size = img_size
        self.mu = mu
        self.density = density
        self.num_targets_reached = 0
        self.max_blocks = max_blocks
        self.block_list = [] # Blocks(self)
        self.add_block(Floor(xlim=self.xlim))
        self.current_step  = 0
        self.num_faces = 4

        self.min_block_reach_target = min_block_reach_target
        self.target_reward_per_block = target_reward_per_block
        self.not_reached_penalty = not_reached_penalty
        
        self.reward_feature = self.get_reward_features(sigma_x=1, sigma_y=1)

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
        return self.state_feature

    def _render_obstacles(self):
        """Overlay des obstacles sur les canaux 0 et 1."""
        # masque initial
        mask = torch.zeros(self.img_size, dtype=torch.bool)
        for obs in self.task.obstacles:
            obs_feat = render_block_2d(obs, xlim=self.xlim, zlim=self.zlim, img_size=self.img_size)
            mask |= obs_feat.to(torch.bool)
        # on met 1.0 là où il y a un obstacle sur les deux canaux
        self.state_feature[0][mask] = 0.09
        self.state_feature[1][mask] = 0.09

    def _render_goals(self):
        """Overlay des goals (targets) sur les canaux 0 et 1."""
        for target in self.task.targets:
            x = target[0]
            z = target[-1]
            col = round((x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.img_size[1] - 1))
            row = round((self.zlim[1] - z)    / (self.zlim[1] - self.zlim[0]) * (self.img_size[0] - 1))
            # on met 1.0 là où il y a un goal
            self.state_feature[0][row, col] = 1.0
            self.state_feature[1][row, col] = 1.0

    def get_reward_features(self, sigma_x = 1, sigma_y = 1):
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
                reward_features += 1 * gaussian((x,y), self.xlim, self.zlim, sigma_x, sigma_y, self.img_size)
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

    def step(self, action : Action, number_step):
        # create and add block to environment
        new_block = self.create_block(action)
        if self.collision(new_block):
            return None, torch.tensor(0.), True
        
        self.add_block(new_block)
        if not self.is_stable():
            return None, torch.tensor(0.), True
        
        
        action_feature = render_block_2d(
            new_block, 
            xlim=self.xlim, 
            zlim=self.zlim, 
            img_size=self.img_size
        )
        
        mask = action_feature.to(torch.bool)      # shape (H,W)
       
        # keep this no matter what print(f"Mask non-zero pixels: {mask.sum().item()}")

       
        # assign block/face values in state_feature
        block_idx = len(self.block_list) - 1
    
        # on réserve [0.0,0.5] pour les blocs, [0.5,1.0] pour les faces
        block_val = 0.2 + (block_idx + 1) / float(self.max_blocks) * 0.8
        face_val  = 0.1 + (action.face + 1) / float(self.num_faces) * 0.1

        self.state_feature[0][mask] = block_val
        self.state_feature[1][mask] = face_val


        self.current_step  += 1

        reward = 0

        not_reached = 0

        for target in self.task.targets:
            if new_block.contains_2d(target):
                self.num_targets_reached += 1
                if len(self.block_list) - self.min_block_reach_target > 0:
                    reward += self.target_reward_per_block  * (len(self.block_list) - self.min_block_reach_target)
            else:
                #if the goal is not reached, we add a penalty
                # note : this is experimental remove i am trying to see if it works
                if len(self.block_list) -1 == self.max_blocks:
                    not_reached -= self.not_reached_penalty


        reward = torch.sum(action_feature * self.reward_feature, dim=(-1, -2)).flatten()[0] + not_reached
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
        elif action.target_face == 0 :
                return True, 'trapezoid_big'
        else:
            return True, 'trapezoid_small'


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
    def __init__(
        self, 
        task, 
        max_blocks      = 6, 
        xlim            = (-5, 5), 
        zlim            = (0, 10), 
        img_size        = (64, 64), 
        mu              = 0.8, 
        density         = 1.0, 
        valid_shapes    = (1, 5),
        n_offsets       = 10,
        limit_steps     = 200,
        target_reward_per_block = 1.0,
        min_block_reach_target = 1,
        collision_penalty = 0.5,
        unstable_penalty = 0.5,
        not_reached_penalty = 25,
        n_floor         = 2
    ):
        # 1) Configuration principale
        self.env = AssemblyEnv(
            task,
            min_block_reach_target,
            target_reward_per_block,
            max_blocks,
            xlim,
            zlim,
            img_size,
            mu,
            density,
            not_reached_penalty
        )
        self.valid_shapes  = list(valid_shapes)
        self.n_offsets     = n_offsets
        self.limit_steps   = limit_steps
        self.n_floor       = n_floor
        # penalties for invalid actions
        self.collision_penalty = collision_penalty
        self.unstable_penalty = unstable_penalty

        # 2) Espaces Gym
        self._init_action_space()
        self._init_observation_space(img_size)

        # 3) Compteurs et états
        self._init_counters()
        # on peut directement reset() si on préfère
        # self.reset()

    def _init_counters(self):
        """Initialise tous les compteurs partagés par reset() et __init__."""
        self.steps           = 0
        self.minus           = 0
        self.negative_reward = 0.0
        self.current_reward = 0

    def reset(self, seed=None, obstacles=None):
        # Set the random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Reinitialize counters
        self._init_counters()

        # Reinitialize the internal environment
        self.env.reset()
        obs = self.env.state_feature
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()  # Convert PyTorch tensor to NumPy array
        return obs, {}
    
    def _init_action_space(self):
        # target_block ∈ [0 .. max_blocks-1], target_face ∈ [0..3], shape_idx ∈ [0..len-1], face ∈ [0..3], offset_idx ∈ [0..n_offsets-1]
        self.action_space = spaces.MultiDiscrete([
            self.env.max_blocks,
            self.env.num_faces,
            len(self.valid_shapes),
            self.env.num_faces,
            self.n_offsets
        ])

    def _init_observation_space(self, img_size):
        C, H, W = 2, *img_size
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(C, H, W),
            dtype=np.float32
        )

    def preprocessing_step(self, action_array):
         #get the number of blocks already in the environment
        self.minus = len(self.env.block_list) - self.n_floor

        # action_array: [target_block, target_face, shape_idx, face, offset_x]
        if action_array is None:
            raise ValueError("Action cannot be None")

        tb, tf, si, fa, off_idx = action_array

        # 0) Forçage n_floor premier blocs au sol
        if self.minus - 1 < 0:
            tb = 0

        # 1) Forçage sol
        if tb == 0:
            tf = 0

        # 2) Conversion de l’index de la forme en ID de la forme
        shape_id = self.valid_shapes[si]

        # 3) Conversion de l’offset discret en valeur réelle
        offset_values = np.linspace(-0.9, 0.9, self.n_offsets)
        ox = float(offset_values[off_idx])

        #enlever si test model brigge v1
        if tb == 0:
            ox *= 4

        # 4) Création de l’action
        action_obj = Action(target_block=tb, target_face=tf, shape=shape_id, face=fa, offset_x=ox)

        # 5) Vérification du bloc cible pour adapter l’offset
        valid, block_type = self.env.get_block(action_obj)
        if valid:
            if block_type == 'cube':
                action_obj.offset_x *= 1.0
            elif block_type == 'trapezoid_big':
                action_obj.offset_x *= 1.5
            elif block_type == 'trapezoid_small':
                action_obj.offset_x *= 1.0

        return action_obj, tb, shape_id,
    
    def get_action_infos(self, action_obj, target_block, shape_id):

        #keep target_reward
        #target_reward = self.env.get_target_reward(action_obj)
        
        action_obj, cause = self.env.test(action_obj)
        info = {
            'num_blocks': len(self.env.block_list)-1,
            'num_targets_reached': 0,
            'cause': cause,
            'stack': 'none',
            'block': shape_id,
            'validate_floor_reward_after_n_floor': 0,
            'validate_floor_reward_before_n_floor': 0,
            'steps': self.steps,
            'truncated': False,
            'current_reward': self.current_reward,
        }
        # Invalid action
        
        self.negative_reward = -1
        if action_obj is None:
            obs = self.env.state_feature.numpy()
            if info['cause'] == 'collision':
                self.negative_reward -= self.collision_penalty
            if info['cause'] == 'unstable':
                self.negative_reward -= self.unstable_penalty
            if target_block != 0:
                info['stack'] = 'try'
            else:
                info['stack'] = 'not_try'
            if self.steps > self.limit_steps:
                info['truncated'] = True
                return obs, -200, False, True, info
            self.current_reward += self.negative_reward
            return obs, self.negative_reward, False, False, info
        else:
            # Valid action

            # Call the step function of the internal environment
            obs, reward, done = self.env.step(action_obj, self.steps)

            # to boost the reward for trapezoid
            if shape_id == 5 and reward > 0:
                reward *= 2

            if isinstance(obs, torch.Tensor):
                obs = obs.numpy()  # Convert PyTorch tensor to NumPy array
            if action_obj.target_block != 0: 
                info['stack'] = 'done'
            else:
                info['stack'] = 'not_done'
                if self.minus >= 0:
                    info['validate_floor_reward_after_n_floor'] = reward
                else:
                    info['validate_floor_reward_before_n_floor'] = reward
            if done:
                info['steps'] = self.steps
            info['num_targets_reached'] = self.env.num_targets_reached
        
        self.current_reward += reward
        return obs, reward, done, False, info

    def step(self, action_array):
        """
        Gym step using get_action_infos to validate actions and safely call inner env.
        """
        self.steps += 1
        self.negative_reward = 0.0
        # Build and preprocess action
        action_obj, target_block, shape_id = self.preprocessing_step(action_array)
        # Validate and step environment
        obs, reward, done, truncated, info = self.get_action_infos(action_obj, target_block, shape_id)
        # Ensure truncation flag
        if not truncated:
            truncated = self.steps >= self.limit_steps
        # Convert tensor obs to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()
        return obs, reward, done, truncated, info
    
    def get_target_reward(self, action):
        return self.env.get_target_reward(action)
    
    def return_env(self):
        """Return the internal AssemblyEnv instance for visualization"""
        return self.env

    def seed(self, seed=None):
        """
        Seed both the internal AssemblyEnv and the Gym spaces.
        """
        # forward seed to internal env if it exists
        if hasattr(self.env, "seed"):
            try:
                self.env.seed(seed)
            except TypeError:
                # gym.Env.seed signature changed in gymnasium
                self.env.seed(seed)
        # seed action & obs spaces
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

