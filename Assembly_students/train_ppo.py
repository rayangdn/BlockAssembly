import gymnasium as gym
import numpy as np
import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecTransposeImage,
)
from wandb.integration.sb3 import WandbCallback

from assembly_rl.environment.tasks import Bridge
from assembly_rl.gym_env import AssemblyGymWrapper


def lr_schedule(progress_remaining: float) -> float:
    """Linearly decay the learning rate from 3e‑4 → 0."""
    return 3e-4 * progress_remaining


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def make_env():
    """Factory that creates one environment instance.

    This *must* be top‑level to be picklable by `SubprocVecEnv`.
    """

    def _init() -> gym.Env:
        task = Bridge(num_stories=2)
        env = AssemblyGymWrapper(task)
        env = ActionMasker(env, mask_fn)  # adds .action_mask attribute
        return env

    return _init


def main():
    config = {
        "algo": "MaskablePPO",
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 4,
        "lr": 1e-4,
        "ent_coef": 1e-3,
        "clip_range": 0.1,
        "total_timesteps": 1_000_000,
        "num_envs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "ent_coef": 1e-3,
    }

    env = SubprocVecEnv([make_env() for _ in range(config["num_envs"])])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = MaskablePPO(
        "CnnPolicy",
        env,
        learning_rate=config["lr"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        verbose=1,
        policy_kwargs=dict(normalize_images=False),
    )

    wandb_callback = None

    model.learn(
        total_timesteps=5000, progress_bar=True, callback=wandb_callback
    )

    model.save("maskable_ppo_bridge")


if __name__ == "__main__":
    main()
