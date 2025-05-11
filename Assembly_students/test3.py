from multiprocessing import freeze_support
import numpy as np
import matplotlib.pyplot as plt
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from assembly_env_copy import AssemblyGymEnv
from tasks import Bridge
from rendering import plot_assembly_env

if __name__ == "__main__":
    freeze_support()
    # Load configuration from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    env_cfg = config["env"]
    task_cfg = config["task"]
    import tasks
    # Build task from config (dynamic type and params)
    task_type = task_cfg["type"]
    task_fn = getattr(tasks, task_type)
    task_args = {k: v for k, v in task_cfg.items() if k != "type"}
    if task_type == "DoubleBridgeStackedTest":
        task = task_fn()
    else:
        task = task_fn(**task_args) if task_args else task_fn()
    # Pass only relevant env params to AssemblyGymEnv
    env_params = {k: v for k, v in env_cfg.items() if k in [
        "max_blocks", "xlim", "zlim", "img_size", "mu", "density", "valid_shapes", "n_offsets", "limit_steps", "n_floor", "target_reward_per_block", "min_block_reach_target", "collision_penalty", "unstable_penalty", "not_reached_penalty"
    ]}
    env = Monitor(AssemblyGymEnv(task=task, **env_params))

    # Load trained model
    model_path = '/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/logs/checkpoints/ppo_block_200000_steps.zip'
    model = PPO.load(model_path)

    # Test loop
    obs, info = env.reset()
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"last{reward}")
            break

    # Visualization and state extraction
    gym_env = env.env
    inner_env = gym_env.return_env()
    block_map, face_map = inner_env.state_feature.numpy()
    print(f"Total reward: {total_reward}")
    assembly_env = inner_env
    plot_assembly_env(
        assembly_env,
        fig=None, ax=None,
        plot_forces=False, force_scale=1.0,
        plot_edges=False, equal=True,
        face_numbers=True, nodes=False,
        task=task
    )
    plt.axis('equal')
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Block ID map")
    plt.imshow(block_map, cmap="tab20")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Face ID map")
    plt.imshow(face_map, cmap="tab10")
    plt.axis("off")
    plt.tight_layout()
    plt.show()