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

# Change gymnasium back to gym
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

    def __hash__(self):
        return hash((self.target_block, self.target_face, self.shape, self.face, self.offset_x))
    

def gaussian(loc, xlim, zlim, sigma_x, sigma_y, img_size=(64, 64)):
    x, y = loc
    X, Y = np.meshgrid(np.linspace(*xlim, img_size[0]), np.linspace(zlim[1], zlim[0], img_size[1]))
    # Only keep values where Y <= y (i.e., below the point)
    mask = (Y <= y)
    gauss = np.exp(-(((X - x)**2 / (2 * sigma_x**2)) + ((Y - y)**2 / (2 * sigma_y**2))))
    gauss = gauss * mask  # Zero out values above the point
    return gauss.reshape(img_size)

class AssemblyEnv(CRA_Assembly):

    def __init__(self, task, min_block_reach_target, target_reward_per_block, end_reward, max_blocks=6,
                 xlim=(-5, 5), zlim=(0, 6), img_size=(64, 64), mu=0.8, density=1.0, not_reached_penalty=25, verbose=False):
        if verbose:
            print("[AssemblyEnv] __init__ arguments:")
            print(f"  task={task}")
            print(f"  min_block_reach_target={min_block_reach_target}")
            print(f"  target_reward_per_block={target_reward_per_block}")
            print(f"  end_reward={end_reward}")
            print(f"  max_blocks={max_blocks}")
            print(f"  xlim={xlim}")
            print(f"  zlim={zlim}")
            print(f"  img_size={img_size}")
            print(f"  mu={mu}")
            print(f"  density={density}")
            print(f"  not_reached_penalty={not_reached_penalty}")
        super().__init__()
        self.task = task
        self.xlim = xlim
        self.zlim = zlim
        self.img_size = img_size
        self.mu = mu
        self.density = density
        self.max_blocks = max_blocks
        self.end_reward = end_reward
        self.not_reached_penalty = not_reached_penalty
        self.num_faces = 4
        self.num_targets_reached = 0
        self.min_block_reach_target = min_block_reach_target
        self.target_reward_per_block = target_reward_per_block

        # Nombre de canaux: un par bloc + obstacles + objectifs + coord_x + coord_z
        C = self.max_blocks + 4
        H, W = self.img_size
        self.state_feature = torch.zeros((C, H, W), dtype=torch.float32)

        # Pré-calcul des canaux de coordonnées absolues
        xs = torch.linspace(self.xlim[0], self.xlim[1], W)
        zs = torch.linspace(self.zlim[1], self.zlim[0], H)
        x_grid = xs.unsqueeze(0).repeat(H, 1)
        z_grid = zs.unsqueeze(1).repeat(1, W)
        x_norm = (x_grid - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        z_norm = (z_grid - self.zlim[0]) / (self.zlim[1] - self.zlim[0])
        self.coord_x = x_norm
        self.coord_z = z_norm

        # Remplissage initial des canaux de coordonnées
        self.state_feature[-2, :, :] = self.coord_x
        self.state_feature[-1, :, :] = self.coord_z

        # Initialise blocs et affichage initial
        self.block_list = []
        self.add_block(Floor(xlim=self.xlim))
        self._render_obstacles()
        self._render_goals()
        self.reward_feature = self.get_reward_features()

    def reset(self, obstacles=None):
        # Réinitialise l'environnement
        self.delete_blocks()
        self.graph._max_node = -1
        self.block_list = []
        self.add_block(Floor(xlim=self.xlim))
        self.num_targets_reached = 0

        # Réinitialisation du tenseur d'état avec rétablissement des canaux de coordonnées
        C = self.max_blocks + 4
        H, W = self.img_size
        self.state_feature = torch.zeros((C, H, W), dtype=torch.float32)
        self.state_feature[-2, :, :] = self.coord_x
        self.state_feature[-1, :, :] = self.coord_z

        # Ré-affichage des obstacles et des objectifs
        self._render_obstacles()
        self._render_goals()
        return self.state_feature

    def _render_obstacles(self):
        """Canal obstacles = index max_blocks"""
        mask = torch.zeros(self.img_size, dtype=torch.bool)
        for obs in self.task.obstacles:
            obs_feat = render_block_2d(obs, xlim=self.xlim, zlim=self.zlim, img_size=self.img_size)
            mask |= obs_feat.to(torch.bool)
        # Canal obstacles déplacé à -4
        self.state_feature[-4][mask] = 1.0

    def _render_goals(self):
        """Canal objectifs = index max_blocks+1"""
        for target in self.task.targets:
            x, z = target[0], target[-1]
            col = round((x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.img_size[1] - 1))
            row = round((self.zlim[1] - z) / (self.zlim[1] - self.zlim[0]) * (self.img_size[0] - 1))
            # Canal objectifs déplacé à -3
            self.state_feature[-3, row, col] = 1.0

    def get_reward_features(self, sigma_x = 1, sigma_y = 2):
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
                reward_features += 0.1 * gaussian((x,y), self.xlim, self.zlim, sigma_x, sigma_y, self.img_size)
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

        new_block = self.create_block(action)
        if self.collision(new_block) or not self.is_stable():
            return None, 0, True, False, {}

        self.add_block(new_block)

        action_feature = render_block_2d(new_block, xlim=self.xlim, zlim=self.zlim, img_size=self.img_size)

        # Update the state feature with the new block
        mask = action_feature.to(torch.bool)      

        block_idx = len(self.block_list) - 2 

        # Update the state feature with the new block depending on the block orientation 
        face = action.face
        face_values = [0.25, 0.5, 0.75, 1.0]
        self.state_feature[block_idx][mask] = face_values[face]

        reward = 0
        #not_reached = 0
        for target in self.task.targets:
            if new_block.contains_2d(target):
                self.num_targets_reached += 1
                


        reward = torch.sum(action_feature * self.reward_feature, dim=(-1, -2)).flatten()[0] #+ not_reached
        reward = float(reward)
        terminated = (len(self.block_list)-1 >= self.max_blocks) | self.num_targets_reached == len(self.task.targets)
        if terminated:
            if self.num_targets_reached == len(self.task.targets) and len(self.block_list)-1 == self.max_blocks:
                #print("all targets reached with max_blocks", len(self.block_list)-1)
                # add a reward if all the targets are reached with the max_blocks number of blocks
                reward += self.end_reward
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
        # Check if the target block is valid
        if action.target_block >= len(self.block_list):
            return None, 'invalid_target_block'
        elif action.target_block == 0 and action.target_face != 0:
            return None, 'invalid_target_face'

        # Create the new block to get collision and stability
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
        
        # Check if the valid action target block is the floor
        if action.target_block == 0:
            return action, 'floor'
        return action, 'good'
    


class AssemblyGymEnv(gym.Env):
    """
    Gymnasium wrapper for the Assembly Environment
    """
    def __init__(
        self, 
        task, 
        max_blocks      = 6, 
        xlim            = (-5, 5), 
        zlim            = (0, 6), 
        img_size        = (64, 64), 
        mu              = 0.8, 
        density         = 1.0, 
        valid_shapes    = (1, 5),
        n_offsets       = 5,
        limit_steps     = 200,
        target_reward_per_block = 1.0,
        min_block_reach_target = 1,
        collision_penalty = 0.5,
        unstable_penalty = 0.5,
        not_reached_penalty = 25,
        n_floor         = 2,
        end_reward = 0,
        verbose=False,
    ):
        if verbose:
            print("[AssemblyGymEnv] __init__ arguments:")
            print(f"  task={task}")
            print(f"  max_blocks={max_blocks}")
            print(f"  xlim={xlim}")
            print(f"  zlim={zlim}")
            print(f"  img_size={img_size}")
            print(f"  mu={mu}")
            print(f"  density={density}")
            print(f"  valid_shapes={valid_shapes}")
            print(f"  n_offsets={n_offsets}")
            print(f"  limit_steps={limit_steps}")
            print(f"  target_reward_per_block={target_reward_per_block}")
            print(f"  min_block_reach_target={min_block_reach_target}")
            print(f"  collision_penalty={collision_penalty}")
            print(f"  unstable_penalty={unstable_penalty}")
            print(f"  not_reached_penalty={not_reached_penalty}")
            print(f"  n_floor={n_floor}")
            print(f"  end_reward={end_reward}")
        super().__init__()
        self.env = AssemblyEnv(
            task,
            min_block_reach_target,
            target_reward_per_block,
            end_reward,
            max_blocks,
            xlim,
            zlim,
            img_size,
            mu,
            density,
            not_reached_penalty,
            verbose=verbose
        )
        self.valid_shapes  = list(valid_shapes)
        self.n_offsets     = n_offsets
        self.limit_steps   = limit_steps
        self.n_floor       = n_floor
        self.end_reward    = end_reward
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
    
    def _init_observation_space(self, img_size):
        C, H, W = self.env.max_blocks + 4, *img_size
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(C, H, W),
            dtype=np.float32
        )

    def _init_counters(self):
        """Initialise tous les compteurs partagés par reset() et __init__."""
        self.steps           = 0
        self.minus           = 0
        self.negative_reward = 0.0
        self.current_reward = 0
        self.true_limit_steps = self.limit_steps
    
    

    def reset(self, *, seed=None, options=None):
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
        if hasattr(obs, 'numpy'):
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


    def preprocessing_step(self, action_array):

        self.steps += 1

        self.negative_reward = 0.0

         #get the number of blocks already in the environment
        #self.minus = len(self.env.block_list) - self.n_floor -1

        if action_array is None:
            raise ValueError("Action cannot be None")

        #1) target_block, target_face, shape_idx, face, offset_x = action_array
        tb, tf, si, fa, off_idx = action_array


        # 2) Conversion de l’index de la forme en ID de la forme
        shape_id = self.valid_shapes[si]

        # 3) Conversion de l’offset discret en valeur réelle
        offset_values = np.linspace(-0.9, 0.9, self.n_offsets)
        ox = float(offset_values[off_idx])

        if tb == 0:
            if ox < 0:
                ox = -ox
            ox *= 2

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
    
        # 6) Vérification de la validité de l’action
        new_action_obj, cause = self.env.test(action_obj)
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

        # -------------------------- Invalid action-------------------------


        self.negative_reward = -1
        if new_action_obj is None:
            obs = self.env.state_feature.numpy()
            if info['cause'] == 'collision':
                self.negative_reward -= self.collision_penalty
            if info['cause'] == 'unstable':
                self.negative_reward -= self.unstable_penalty
            if info['cause'] == 'invalid_target_block':
                self.negative_reward -= 0
            if tb != 0:
                info['stack'] = 'try'
            else:
                info['stack'] = 'not_try'
            if self.steps > self.limit_steps:
                info['truncated'] = True
                self.negative_reward -= 10
                '''print("Truncated with")
                print("len(block_list)", len(self.env.block_list) - 1, self.negative_reward)'''
                return obs, self.negative_reward, False, True, info
            return obs, self.negative_reward, False, False, info
        else:
            # ------------------------- Valid action-------------------------
            self.true_limit_steps = self.steps + self.limit_steps
            #print("limit_steps --", self.true_limit_steps)

            # Call the step function of the internal environment
            obs, reward, done  = self.env.step(new_action_obj, self.steps)

            if hasattr(obs, 'numpy'):
                obs = obs.numpy()  # Convert PyTorch tensor to NumPy array for Gym

            # get information about the action
            if new_action_obj.target_block != 0: 
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
        print("good action with", reward, len(self.env.block_list) - 1, self.steps, self.env.num_targets_reached)

        # give -1 to force the agent to get close to the target
        return obs, reward - 1, done, False, info

    def step(self, action):
        """
        Gym step using get_action_infos to validate actions and safely call inner env.
        """
        '''if len(self.env.block_list) - 1 >= self.env.max_blocks:
            print("max_blocks2", len(self.env.block_list) - 1, self.end_reward)
            return self.env.state_feature, 20, True, False, {}'''
    
        # Build and preprocess action
        obs, reward, done, truncated, info = self.preprocessing_step(action)
        # Convert tensor obs to numpy
        if hasattr(obs, 'numpy'):
            obs = obs.numpy()

        return obs, reward, done, truncated, info

    
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

