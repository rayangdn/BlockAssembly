from multiprocessing import freeze_support
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from assembly_env_copy import AssemblyGymEnv
from tasks import Bridge
from rendering import plot_assembly_env

if __name__ == "__main__":
    freeze_support()
    # ——— Configuration ———
    #model_path = '/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/logs/best_model_cont/best_model.zip'
    model_path = '/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/logs/best_model/best_model.zip'
    #model_path = '/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/logs/checkpoints/ppo_block_120000_steps.zip'
    #model_path = '/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/logs/checkpoints_cont/ppo_block_continue_56000_steps.zip'
    # ——— Create and wrap env ———
    task = Bridge(num_stories=1, width=2)
    env = Monitor(AssemblyGymEnv(task=task))


    # ——— Load trained model ———
    model = PPO.load(model_path)

    # ——— Test loop ———
    obs, info = env.reset()
    total_reward = 0.0

    # Gymnasium new API: step → obs, reward, terminated, truncated, info
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"last{reward}")
            break

    # 1) Récupérer le Monitor → AssemblyGymEnv
    gym_env = env.env   # AssemblyGymEnv

    # 2) Descendre au AssemblyEnv interne
    inner_env = gym_env.return_env()         # AssemblyEnv

    # 3) Extraire state_feature
    block_map, face_map = inner_env.state_feature.numpy()

    print(f"Total reward: {total_reward}") 

    # ——— Visualization ———
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

    # Show final observation images
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