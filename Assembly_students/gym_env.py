from itertools import product

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces

from assembly_env import Action, AssemblyEnv
from rendering import plot_assembly_env


class AssemblyGymWrapper(gym.Env):
    def __init__(self, task, overlap=0.2, render=False):
        super().__init__()
        self._render = render

        self.task = task
        self.env = AssemblyEnv(task)
        self.overlap = overlap

        self.max_blocks = self.env.max_blocks
        self.faces_per_block = 4
        self.num_shapes = len(task.shapes)
        self.num_offsets = len(task.floor_positions)
        self.offset_fracs = np.linspace(-1.0, 1.0, self.num_offsets)
        self.shape_map = {
            s.block_id: idx for idx, s in enumerate(self.task.shapes)
        }

        self.all_actions = list(
            product(
                range(self.max_blocks),
                range(self.faces_per_block),
                range(self.num_shapes),
                range(self.faces_per_block),
                range(self.num_offsets),
            )
        )

        self.action_tuple_to_idx = {
            t: i for i, t in enumerate(self.all_actions)
        }

        self.action_space = spaces.Discrete(len(self.all_actions))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, *self.env.state_feature.shape),
            dtype=np.uint8,
        )

    def reset(self, *, seed=None, options=None):
        self.env.reset()
        obs = (
            (self.env.state_feature.unsqueeze(0) * 255)
            .numpy()
            .astype(np.uint8)
        )
        return obs, {}

    def _decode_action(self, action_tuple):
        block_idx, tgt_face, shape_idx, face, frac_idx = action_tuple

        block = self.env.block_list[block_idx]
        shape = self.task.shapes[shape_idx]

        frac = self.offset_fracs[frac_idx]
        l1 = block.face_length_2d(tgt_face) / 2
        l2 = shape.face_length_2d(face) / 2
        max_range = (l1 + l2) * (1 - self.overlap)
        offset_x = frac * max_range

        return Action(block_idx, tgt_face, shape.block_id, face, offset_x)

    def step(self, action):
        action_tuple = self.all_actions[action]
        act = self._decode_action(action_tuple)
        if act.target_block >= len(self.env.block_list):
            return (
                self.env.state_feature.unsqueeze(0).numpy(),
                -1.0,
                True,
                False,
                {},
            )

        if self._render:
            plot_assembly_env(self.env, task=self.task)
            plt.axis("equal")
            plt.show()

        obs, reward, done = self.env.step(act)
        if obs is None:
            obs = self.env.state_feature
        obs = (obs.unsqueeze(0) * 255).numpy().astype(np.uint8)

        return obs, reward.item(), done, False, {}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)

        for offset_idx, action in self.env.available_actions(
            num_block_offsets=self.num_offsets, overlap=self.overlap
        ):
            # if self.env.is_valid_action(action):
            key = (
                action.target_block,
                action.target_face,
                self.shape_map[action.shape],
                action.face,
                offset_idx,
            )
            mask[self.action_tuple_to_idx[key]] = True

        return mask

    # For testing purposes
    def valid_action_sampler(self):
        shape_map = {s.block_id: idx for idx, s in enumerate(self.task.shapes)}

        action = self.env.random_action()
        if action is None:
            return None

        action_tuple = (
            action.target_block,
            action.target_face,
            shape_map[action.shape],
            action.face,
            0,
        )
        return self.action_tuple_to_idx[action_tuple]
