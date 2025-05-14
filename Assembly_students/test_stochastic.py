import torch
import yaml
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from assembly_env_copy import AssemblyGymEnv
from rendering import plot_assembly_env
from custom_cnn import CustomCNN
from tasks import StochasticBridge
import gymnasium as gym
import numpy as np
import os

# x_offsets as in stochastic_train.py
x_offsets = [-1.75, -0.825, 0.0, 0.825, 1.75]

def make_env(config, x_pos=0.0, use_timelimit=True, max_episode_steps=200):
    env_cfg = config["env"]
    env_param_keys = [
        "max_blocks", "xlim", "zlim", "img_size", "mu", "density", "valid_shapes", "n_offsets", "limit_steps", "n_floor",
        "target_reward_per_block", "min_block_reach_target", "collision_penalty", "unstable_penalty", "not_reached_penalty", "end_reward"
    ]
    env_params = {k: v for k, v in env_cfg.items() if k in env_param_keys}
    task = StochasticBridge(x_pos=x_pos)
    env = AssemblyGymEnv(task=task, **env_params, verbose=False)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if use_timelimit:
        max_blocks = env_params.get('max_blocks', 3)
        max_episode_steps = max_blocks + 1
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env, task

def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Model path as in stochastic_train.py
    #model_path = config.get('output_model_path', 'ppo_checkpoints/ppo_stochastic_env.zip')
    model_path = '/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/logs/best_model/best_model.zip'
    if not os.path.exists(model_path):
        model_path = "logs/best_model/best_model.zip"

    # Use x_pos=0.0 for default, but allow testing all x_offsets
    x_pos_list = x_offsets
    n_eval_episodes = 3

    # Device logic as in stochastic_train.py
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load model ONCE
    # Create a dummy env for loading
    dummy_env, _ = make_env(config, x_pos=0.0, use_timelimit=True)
    obs_shape = dummy_env.observation_space.shape
    img_size = tuple(obs_shape[1:])
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(img_size=img_size),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    model = PPO.load(
        model_path,
        env=dummy_env,
        device=device,
        policy_kwargs=policy_kwargs
    )

    for x_pos in x_pos_list:
        print(f"\n=== Evaluation for x_pos={x_pos} ===")
        env, task = make_env(config, x_pos=x_pos, use_timelimit=True)
        model.set_env(env)
        # Deterministic evaluation
        for ep in range(n_eval_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            print(f"[Deterministic x_pos={x_pos} ep={ep+1}] Total reward: {total_reward}")
            # Unwrap to base env for plotting
            base_env = env
            while hasattr(base_env, "env"):
                base_env = base_env.env
            plot_assembly_env(base_env, task=task)
            plt.axis('equal')
            plt.savefig(f"plot_det_x{x_pos}_ep{ep+1}.png")
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
            print(f"[Stochastic x_pos={x_pos} ep={ep+1}] Total reward: {total_reward}")
            base_env = env
            while hasattr(base_env, "env"):
                base_env = base_env.env
            plot_assembly_env(base_env, task=task)
            plt.axis('equal')
            plt.savefig(f"plot_stoch_x{x_pos}_ep{ep+1}.png")
            plt.close()

if __name__ == "__main__":
    main()
