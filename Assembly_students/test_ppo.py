import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.evaluation import evaluate_policy

from tasks import Bridge
from gym_env import AssemblyGymWrapper


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


task = Bridge(num_stories=2)
env = AssemblyGymWrapper(task, render=True)
env = ActionMasker(env, mask_fn)

# (2) Load the model, passing in the env for correct masking:
model = MaskablePPO.load("maskable_ppo_bridge_fin", env=env)

terminated = False
obs, _ = env.reset()
while not terminated:
    # Retrieve current action mask
    action_masks = mask_fn(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)

print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
