import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from tasks import Bridge
from assembly_env_copy import AssemblyGymEnv
from cause_logging_callback import CauseLoggingCallback


def make_env(rank: int, seed: int = 0):
    """
    Helper to create a monitored environment with a fixed seed.
    """
    def _init():
        task = Bridge(num_stories=2)
        env = AssemblyGymEnv(task=task)
        env.seed(seed + rank)
        return Monitor(env)
    set_random_seed(seed + rank)
    return _init


def main():
    # 1) Set global seeds for reproducibility
    seed = 42
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2) Create parallel training environments
    n_envs = 8
    train_envs = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])

    # 3) Create evaluation environment
    eval_env = DummyVecEnv([make_env(0, seed + 100)])

    # 4) Configure callbacks for checkpointing and evaluation
    os.makedirs("logs/checkpoints_cont", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=1000,
        save_path="logs/checkpoints_cont",
        name_prefix="ppo_block_continue"
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="logs/best_model_cont",
        log_path="logs/eval_results_cont",
        eval_freq=5_000 // n_envs,
        deterministic=True,
        render=False
    )
    cause_cb = CauseLoggingCallback()

    # 5) Choose device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # 6) Load the previously trained model
    model_path = "ppo_checkpoints/ppo_raw_env.zip"
    model = PPO.load(model_path, env=train_envs, device=device)

    # 7) Continue training for additional timesteps
    additional_timesteps = 200_000  # Adjust this as needed
    model.learn(
        total_timesteps=additional_timesteps,
        callback=[checkpoint_cb, eval_cb, cause_cb]
    )

    # 8) Save the continued model
    os.makedirs("ppo_checkpoints_continue", exist_ok=True)
    model.save("ppo_checkpoints_continue/ppo_continued.zip")


if __name__ == "__main__":
    main()
