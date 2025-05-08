from multiprocessing import freeze_support

import numpy as np

import gymnasium as gym
from assembly_env import AssemblyGymEnv
from tasks import Bridge
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import matplotlib.pyplot as plt
from rendering import plot_assembly_env, plot_task
from tasks import Bridge
from blocks import Floor



# Load the checkpoint
model_path = '/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/ppo_checkpoints/ppo_assembly_160000_steps.zip'
model = PPO.load(model_path)

task = Bridge(num_stories=1)

# Use AssemblyGymEnv instead of AssemblyEnv
env = AssemblyGymEnv(task = task)
obs, _ = env.reset()  # Get initial observation (gym returns info as second value)
done = False
print(type(obs), obs)
print(obs.shape)
rewards = 0
terminated = False
truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=False)  # Use the model to predict the action

    if action is None:
        break
    print(action)
    obs, r, terminated, truncated, info = env.step(action) 
    print(info) # Gym env returns 5 values
    rewards += r
    
    # If you want to check stability, you can access the underlying environment
    print(f"Stable: {env.env.is_stable()}")
    
print(f"Total reward: {rewards}")

# Get the underlying AssemblyEnv for plotting
assembly_env = env.env
plot_assembly_env(assembly_env, task=task, face_numbers=True)
plt.axis('equal')

block_map, face_map = obs  # shape: (64, 64) from (1, 64, 64)
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Block ID map")
plt.imshow(block_map, cmap="tab20")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Face ID map")
plt.imshow(face_map, cmap="tab10")
plt.axis("off")

plt.tight_layout()
plt.show()

