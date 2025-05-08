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

def make_env():
    def _init():
        env = AssemblyGymEnv(task=Bridge(num_stories=1))
        return Monitor(env)
    return _init

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # ——— Parallel training env ———
    n_envs = 8

    # build your train_vec exactly once
    train_vec = SubprocVecEnv([make_env() for _ in range(n_envs)])
    train_vec = VecNormalize.load(
       "ppo_checkpoints/vecnormalize.pkl",
        train_vec)



        # 2) Load the checkpoint
    #   *SB3 checkpoints are .zip files by default; add the extension if needed.*
    model_path = "/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/ppo_checkpoints/ppo_assembly_80000_steps.zip"
    model = PPO.load(model_path, env=train_vec, device=device)

   

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
        callback=[checkpoint_cb],
    )
    model.save("ppo_assembly_bridge_cnn")

if __name__ == "__main__":
    # on Windows/macOS spawn mode, needed for safety
    freeze_support()
    main()