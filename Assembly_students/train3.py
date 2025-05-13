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
from custom_cnn import CustomCNN
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym  # Use gymnasium for compatibility

# Load training configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# wrapper settings
env_cfg = config["env"]
wrapper_cfg = config["env_wrappers"]
seed = wrapper_cfg["seed"]
n_envs = wrapper_cfg["n_envs"]
eval_envs_count = wrapper_cfg.get("eval_envs", 1)

def make_env(rank: int, use_timelimit=False, max_episode_steps=5):
    def _init():
        # build Task from config
        task_conf = config['task']
        task_fn = getattr(tasks, task_conf['type'])
        if 'args' in task_conf:
            task = task_fn(**task_conf['args'])
        else:
            task = task_fn()
        # Only pass valid env params
        env_param_keys = [
            "max_blocks", "xlim", "zlim", "img_size", "mu", "density", "valid_shapes", "n_offsets", "limit_steps", "n_floor",
            "target_reward_per_block", "min_block_reach_target", "collision_penalty", "unstable_penalty", "not_reached_penalty", "end_reward"
        ]
        env_params = {k: v for k, v in env_cfg.items() if k in env_param_keys}
        env = AssemblyGymEnv(task=task, **env_params, verbose=False)
        env.seed(seed + rank)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # gymnasium Monitor equivalent
        if use_timelimit:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env
    set_random_seed(seed + rank)
    return _init

def main():
    # 1) Seed
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2) Parallel train envs
    train_envs = SubprocVecEnv([make_env(i, use_timelimit=False) for i in range(n_envs)])
    train_envs = VecNormalize(train_envs, norm_obs=False, norm_reward=True, clip_reward=100.0)
    # 3) Eval env (offset seed by +100, with step limit)
    eval_env = DummyVecEnv([make_env(100, use_timelimit=True, max_episode_steps=200)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=100.0)
    eval_env.training = False
    eval_env.norm_reward = False

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
    )
    cause_cb = CauseLoggingCallback()

    # 5) Model
    ppo_cfg = config['ppo']
    device = torch.device(ppo_cfg.get('device', 'cpu')) if ppo_cfg['device'] != 'auto' else (
        torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    )

    env_param_keys = [
        "max_blocks", "xlim", "zlim", "img_size", "mu", "density", "valid_shapes", "n_offsets", "limit_steps", "n_floor",
        "target_reward_per_block", "min_block_reach_target", "collision_penalty", "unstable_penalty", "not_reached_penalty", "end_reward"
    ]
    env_params = {k: v for k, v in env_cfg.items() if k in env_param_keys}

    # Always use custom CNN
    task_conf = config['task']
    task_fn = getattr(tasks, task_conf['type'])
    if 'args' in task_conf:
        dummy_task = task_fn(**task_conf['args'])
    else:
        dummy_task = task_fn()
    max_blocks = env_params.get('max_blocks', 3)  # Get from env_params/env_cfg!

    # Print env_params once for debugging using AssemblyGymEnv's verbose mode
    dummy_env = AssemblyGymEnv(task=dummy_task, **env_params, verbose=True)
    obs_shape = dummy_env.observation_space.shape  # (channels, H, W)
    img_size = tuple(obs_shape[1:])  # (H, W)
    # Only change num_target_blocks and n_offsets in action_dims
    num_target_blocks = max_blocks - 1
    n_offsets = env_params.get('n_offsets', 7)
    action_dims = (num_target_blocks, 4, 2, 4, n_offsets)
    # After printing, set verbose=False for all other envs

    model = PPO(
        'CnnPolicy',
        train_envs,
        learning_rate=float(ppo_cfg['learning_rate']),
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        clip_range=ppo_cfg['clip_range'],
        gamma=ppo_cfg['gamma'],
        ent_coef=ppo_cfg['ent_coef'],
        target_kl=ppo_cfg['target_kl'],
        device=device,
        verbose=1,
        tensorboard_log=ppo_cfg.get('tensorboard_log', './ppo_assembly_tensorboard/'),
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                img_size=img_size,
                action_dims=action_dims,
            )
        )
    )

    # 6) Train
    model.learn(
        total_timesteps=config['ppo'].get('total_timesteps', ppo_cfg['total_timesteps']),
        callback=[checkpoint_cb, eval_cb, cause_cb]
    )

    # 7) Save final model
    model.save(config.get('output_model_path', 'ppo_checkpoints/ppo_raw_env.zip'))

    # 8) Save VecNormalize
    train_envs.training = False  # Stop updating stats during saving or later evaluation
    train_envs.save("logs/vecnormalize.pkl")

if __name__ == "__main__":
    main()
