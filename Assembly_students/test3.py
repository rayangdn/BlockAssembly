from multiprocessing import freeze_support
import numpy as np
import matplotlib.pyplot as plt
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
from assembly_env_copy import AssemblyGymEnv
from tasks import Bridge
from rendering import plot_assembly_env
from custom_cnn import CustomCNN

if __name__ == "__main__":
    freeze_support()
    # Load configuration from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    env_cfg = config["env"]
    task_cfg = config["task"]
    ppo_cfg = config["ppo"]
    checkpoint_cfg = config.get("checkpoint", {})

    import tasks
    # Build task from config (dynamic type and params)
    task_type = task_cfg["type"]
    task_fn = getattr(tasks, task_type)
    task_args = task_cfg.get("args", {})
    task = task_fn(**task_args) if task_args else task_fn()

    # Pass all env params to AssemblyGymEnv (filter only valid keys)
    env_param_keys = [
        "max_blocks", "xlim", "zlim", "img_size", "mu", "density", "valid_shapes", "n_offsets", "limit_steps", "n_floor",
        "target_reward_per_block", "min_block_reach_target", "collision_penalty", "unstable_penalty", "not_reached_penalty", "end_reward"
    ]
    env_params = {k: v for k, v in env_cfg.items() if k in env_param_keys}
    env = AssemblyGymEnv(task=task, **env_params)

    # Load VecNormalize stats if available
    try:
        env = VecNormalize.load("logs/vecnormalize.pkl", env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize statistics.")
    except Exception as e:
        print("VecNormalize stats not loaded:", e)

    # Load trained model ; choose between checkpoint and best model
    model_path = checkpoint_cfg.get("save_path", "logs/checkpoints") + "/ppo_block_376000_steps.zip"
    #model_path ="/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/logs/best_model/best_model.zip"
    model = PPO.load(model_path)

    # --- Deterministic run ---
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    while True and steps < 5:
        print(f"Step {steps}")
        steps += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"[Deterministic] Last reward: {reward}, Total reward: {total_reward}, Steps: {steps}")
            break

    # Visualization for deterministic run
    state_feature = env.env.state_feature.numpy()
    plt.figure(figsize=(12, 4))
    for idx in range(state_feature.shape[0]):
        plt.subplot(1, state_feature.shape[0], idx + 1)
        plt.title(f"Channel {idx}")
        plt.imshow(state_feature[idx], cmap="tab20", vmin=0.0, vmax=1.0)
        plt.axis("off")
    plt.suptitle("Deterministic run: All state feature channels")
    plt.tight_layout()
    plt.show()
    print(f"[Deterministic] Total reward: {total_reward}")
    plot_assembly_env(
        env,
        fig=None, ax=None,
        plot_forces=False, force_scale=1.0,
        plot_edges=False, equal=True,
        face_numbers=True, nodes=False,
        task=task
    )
    plt.axis('equal')
    plt.show()

    # --- 10 stochastic runs ---
    for run in range(10):
        obs, info = env.reset()
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"[Stochastic {run+1}] Last reward: {reward}")
                break

        # Visualization for each stochastic run
        print(f"[Stochastic {run+1}] Total reward: {total_reward}")
        plot_assembly_env(
            env,
            fig=None, ax=None,
            plot_forces=False, force_scale=1.0,
            plot_edges=False, equal=True,
            face_numbers=True, nodes=False,
            task=task
        )
        plt.axis('equal')
        plt.show()