from multiprocessing import freeze_support
import sys, os
import numpy as np
import torch
import gymnasium as gym
from assembly_env import AssemblyGymEnv
from tasks import Bridge
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from stable_baselines3.common.callbacks import BaseCallback

class CauseLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cause_counts = {
            'collision': 0,
            'unstable': 0,
            'invalid_target_block': 0,
            'invalid_target_face_floor': 0,
            'invalid_offset_x_cube': 0,
            'invalid_offset_x_trapezoid_3': 0,
            'other': 0
        }

    def _on_step(self) -> bool:
        # Info dicts are accessible via self.locals
        infos = self.locals.get("infos", [])
        for info in infos:
            cause = info.get("cause", "other")
            if cause in self.cause_counts:
                self.cause_counts[cause] += 1
            else:
                self.cause_counts["other"] += 1
        
        # Log to TensorBoard
        for cause, count in self.cause_counts.items():
            self.logger.record(f"failures/{cause}", count)

        return True

def make_env():
    def _init():
        env = AssemblyGymEnv(task=Bridge(num_stories=1))
        return Monitor(env)
    return _init

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # ——— Parallel training env ———
    n_envs = 8
    train_vec = SubprocVecEnv([make_env() for _ in range(n_envs)])
    train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_reward=10.0)

    os.makedirs("ppo_checkpoints", exist_ok=True)
    train_vec.save("ppo_checkpoints/vecnormalize.pkl")

    # ——— Build & train ———
    model = PPO(
        "CnnPolicy",
        train_vec,
        learning_rate=5e-5,
        n_steps=256,
        batch_size=64,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.005,
        target_kl=0.02,
        policy_kwargs={'normalize_images' : False, "net_arch": dict(pi=[128, 128],
        vf=[256, 256])},
        tensorboard_log="./ppo_assembly_tensorboard/",
        device=device,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path="./ppo_checkpoints/",
        name_prefix="ppo_assembly",
    )
    eval_cb = EvalCallback(
        eval_env=VecNormalize(
            SubprocVecEnv([make_env()]),
            norm_obs=True, norm_reward=True, clip_reward=10.0
        ),
        eval_freq=10_000,
        n_eval_episodes=5,
        log_path="./eval_logs/",
        best_model_save_path="./best_model/",
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=1_000_000,
        log_interval=1,
        callback=[checkpoint_cb, CauseLoggingCallback()]
    )
    model.save("ppo_assembly_bridge_cnn")

if __name__ == "__main__":
    # on Windows/macOS spawn mode, needed for safety
    freeze_support()
    main()