import gymnasium as gym
import numpy as np
import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from wandb.integration.sb3 import WandbCallback

from gym_env import AssemblyGymWrapper
from tasks import Bridge

from stable_baselines3.common.logger import configure

tmp_path = "./log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


use_wandb = True

if use_wandb:
    wandb.init(
        project="block-assembly",
        sync_tensorboard=True,
        monitor_gym=True,
    )

task = Bridge(num_stories=2)
env = AssemblyGymWrapper(task)
env = ActionMasker(env, mask_fn)

model = MaskablePPO("CnnPolicy", env, verbose=1)
model.set_logger(new_logger)
model.learn(
    total_timesteps=5000,
    progress_bar=True,
    callback=(
        WandbCallback(gradient_save_freq=10, verbose=2) if use_wandb else None
    ),
)

model.save("maskable_ppo_bridge")
