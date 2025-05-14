import torch
import yaml
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from assembly_env_copy import AssemblyGymEnv
from rendering import plot_assembly_env
from custom_cnn import CustomCNN
import tasks
import gymnasium as gym
import numpy as np
import os

def make_env(config, rank=0, use_timelimit=False, max_episode_steps=200):
    env_cfg = config["env"]
    task_cfg = config["task"]
    seed = config.get("env_wrappers", {}).get("seed", 0)
    # Build task
    task_type = task_cfg["type"]
    task_fn = getattr(tasks, task_type)
    task_args = task_cfg.get("args", {})
    task = task_fn(**task_args) if task_args else task_fn()
    env_param_keys = [
        "max_blocks", "xlim", "zlim", "img_size", "mu", "density", "valid_shapes", "n_offsets", "limit_steps", "n_floor",
        "target_reward_per_block", "min_block_reach_target", "collision_penalty", "unstable_penalty", "not_reached_penalty", "end_reward"
    ]
    env_params = {k: v for k, v in env_cfg.items() if k in env_param_keys}
    env = AssemblyGymEnv(task=task, **env_params, verbose=False)
    env.seed(seed + rank)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if use_timelimit:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env, task

def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    checkpoint_cfg = config.get("checkpoint", {})
    # Build env and task (NO DummyVecEnv, NO VecNormalize)
    env, task = make_env(config, rank=0, use_timelimit=True, max_episode_steps=200)

    obs_shape = env.observation_space.shape
    img_size = tuple(obs_shape[1:])
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            img_size=img_size,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    #model_path = checkpoint_cfg.get("save_path", "logs/checkpoints") + "/ppo_block_240000_steps.zip"
    model_path = '/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/logs/best_model/best_model.zip'

    if torch.backends.mps.is_available():
        device = torch.device('cuda')
    elif torch.cuda.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = PPO.load(
        model_path,
        env=env,
        device=device,
        policy_kwargs=policy_kwargs
    )
    # Deterministic evaluation
    n_eval_episodes = 3
    for ep in range(1):
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"[Deterministic {ep+1}] Total reward: {total_reward}")
        # Unwrap to base env for plotting
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        plot_assembly_env(base_env, task=task)
        plt.axis('equal')
        plt.savefig(f"plot_det_{ep+1}.png")
        plt.close()
    # Stochastic evaluation
    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"[Stochastic {ep+1}] Total reward: {total_reward}")
        # Unwrap to base env for plotting
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        plot_assembly_env(base_env, task=task)
        plt.axis('equal')
        plt.savefig(f"plot_stoch_{ep+1}.png")
        plt.close()

if __name__ == "__main__":
    main()