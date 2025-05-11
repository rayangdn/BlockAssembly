import os
import torch
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from tasks import Bridge
from assembly_env_copy import AssemblyGymEnv
from cause_logging_callback import CauseLoggingCallback
import tasks

# Load training configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# wrapper settings
env_cfg = config["env"]
wrapper_cfg = config["env_wrappers"]
seed = wrapper_cfg["seed"]
n_envs = wrapper_cfg["n_envs"]
eval_envs_count = wrapper_cfg.get("eval_envs", 1)

def make_env(rank: int):
    def _init():
        # build Task from config
        task_conf = config['task']
        task_fn = getattr(tasks, task_conf['type'])
        # Only pass arguments if the task function accepts them
        if task_conf['type'] == 'DoubleBridgeStackedTest':
            task = task_fn()
        else:
            task_args = {k: v for k, v in task_conf.items() if k != 'type'}
            task = task_fn(**task_args)
        # init environment with configured params
        env = AssemblyGymEnv(task=task, **env_cfg)
        env.seed(seed + rank)
        return Monitor(env)
    set_random_seed(seed + rank)
    return _init

def main():
    # 1) Seed
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2) Parallel train envs
    train_envs = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # 3) Eval env (offset seed by +100)
    eval_env = DummyVecEnv([make_env(100)])

    # 4) Callbacks
    os.makedirs("logs/checkpoints", exist_ok=True)
    ck_cfg = config['checkpoint']
    checkpoint_cb = CheckpointCallback(
        save_freq=ck_cfg['save_freq'],
        save_path=ck_cfg['save_path'],
        name_prefix=ck_cfg['name_prefix']
    )
    ev_cfg = config['evaluation']
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=ev_cfg['best_model_save_path'],
        log_path=ev_cfg['log_path'],
        eval_freq=ev_cfg['eval_freq'],
        deterministic=ev_cfg['deterministic'],
        render=ev_cfg['render']
    )
    cause_cb = CauseLoggingCallback()

    # 5) Model
    ppo_cfg = config['ppo']
    device = torch.device(ppo_cfg.get('device', 'cpu')) if ppo_cfg['device'] != 'auto' else (
        torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    )
    model = PPO(
        ppo_cfg['policy'],
        train_envs,
        learning_rate=float(ppo_cfg['learning_rate']),  # <-- Ensure float type
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        clip_range=ppo_cfg['clip_range'],
        gamma=ppo_cfg['gamma'],
        ent_coef=ppo_cfg['ent_coef'],
        target_kl=ppo_cfg['target_kl'],
        policy_kwargs=ppo_cfg['policy_kwargs'],
        device=device,
        verbose=1,
        tensorboard_log=ppo_cfg.get('tensorboard_log', './ppo_assembly_tensorboard/')
    )

    # 6) Train
    model.learn(
        total_timesteps=config['ppo'].get('total_timesteps', 200_000),
        callback=[checkpoint_cb, eval_cb, cause_cb]
    )

    # 7) Save final model
    model.save(config.get('output_model_path', 'ppo_checkpoints/ppo_raw_env.zip'))

if __name__ == "__main__":
    main()
