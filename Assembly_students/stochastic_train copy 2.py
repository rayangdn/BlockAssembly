import os
import torch
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from tasks import StochasticBridge
from assembly_env_copy_2 import AssemblyGymEnv
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
from custom_cnn import CustomCNN
from cause_logging_callback import CauseLoggingCallback
import datetime

# Load config
with open("config copy 3.yaml", "r") as f:
    config = yaml.safe_load(f)
env_cfg = config["env"]
wrapper_cfg = config["env_wrappers"]
seed = wrapper_cfg["seed"]
n_envs = wrapper_cfg["n_envs"]

x_offsets = [-1.75, -0.825, 0.0, 0.825, 1.75]

def make_weighted_env(proportions, env_params, seed_base):
    def _init():
        x_pos = np.random.choice(x_offsets, p=proportions)
        task = StochasticBridge(x_pos=x_pos)
        env = AssemblyGymEnv(task=task, **env_params, verbose=False)
        env.seed(seed_base + np.random.randint(10000))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _init

def evaluate_model(model, x_pos, env_params, n_eval_episodes=10, deterministic=True):
    env = DummyVecEnv([lambda: AssemblyGymEnv(
        task=StochasticBridge(x_pos=x_pos), **env_params, verbose=False)])
    env = VecNormalize.load("logs_copy2/vecnormalize.pkl", env) if os.path.exists("logs_copy2/vecnormalize.pkl") else env
    env.training = False
    env.norm_reward = False
    rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
        rewards.append(total_reward)
    env.close()
    return np.mean(rewards)

def main():
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_param_keys = [
        "max_blocks", "xlim", "zlim", "img_size", "mu", "density", "valid_shapes", "n_offsets", "limit_steps", "n_floor",
        "target_reward_per_block", "min_block_reach_target", "collision_penalty", "unstable_penalty", "not_reached_penalty", "end_reward"
    ]
    env_params = {k: v for k, v in env_cfg.items() if k in env_param_keys}

    # Initial uniform proportions
    proportions = [1/len(x_offsets)] * len(x_offsets)

    # --- Dummy env for img_size and action_dims ---
    dummy_task = StochasticBridge(x_pos=0.0)
    dummy_env = AssemblyGymEnv(task=dummy_task, **env_params, verbose=True)
    obs_shape = dummy_env.observation_space.shape  # (channels, H, W)
    img_size = tuple(obs_shape[1:])  # (H, W)
    dummy_env.close()
    # ---------------------------------------------

    # PPO/model setup from test3.py
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(img_size=img_size),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # Device logic from test3.py
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create initial training envs
    env_fns = [make_weighted_env(proportions, env_params, seed + i*1000) for i in range(n_envs)]
    train_envs = SubprocVecEnv(env_fns)
    train_envs = VecNormalize(train_envs, norm_obs=False, norm_reward=True, clip_reward=100.0)

    # --- EvalCallback setup: 5 envs, one for each x_pos, with TimeLimit ---
    max_blocks = env_params.get('max_blocks', 3)
    max_episode_steps = max_blocks + 1
    eval_env = DummyVecEnv([
        (lambda x=x: gym.wrappers.TimeLimit(
            AssemblyGymEnv(task=StochasticBridge(x_pos=x), **env_params, verbose=False),
            max_episode_steps=max_episode_steps
        ))
        for x in x_offsets
    ])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=100.0)
    eval_env.training = False
    eval_env.norm_reward = False

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="logs_copy2/best_model",
        log_path="logs_copy2/eval",
        eval_freq=2048,
        deterministic=True,
        n_eval_episodes=5,
    )
    # ------------------------------------------------------

    # Create callback list
    callback_list = CallbackList([eval_cb, CauseLoggingCallback()])

    # Tensorboard log directory setup
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_log_dir = os.path.join(
        config['ppo'].get('tensorboard_log', './ppo_assembly_tensorboard_copy2/'),
        f"run_{run_id}"
    )

    model = PPO(
        policy="CnnPolicy", 
        env=train_envs,
        learning_rate=float(config['ppo']['learning_rate']),
        n_steps=config['ppo']['n_steps'],
        batch_size=config['ppo']['batch_size'],
        n_epochs=config['ppo']['n_epochs'],
        clip_range=config['ppo']['clip_range'],
        gamma=config['ppo']['gamma'],
        ent_coef=config['ppo']['ent_coef'],
        target_kl=config['ppo']['target_kl'],
        device=device,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs
    )

    total_iterations = 50
    steps_per_iter = 8192

    for iteration in range(total_iterations):
        print(f"\n=== Training iteration {iteration+1}/{total_iterations} ===")
        model.learn(total_timesteps=steps_per_iter, reset_num_timesteps=False, callback=callback_list)

        # Save VecNormalize stats
        train_envs.save("logs_copy2/vecnormalize.pkl")

        # Manual evaluation on each x_pos (optional, for detailed stats)
        x_pos_rewards = {}
        for x in x_offsets:
            mean_reward = evaluate_model(model, x, env_params, n_eval_episodes=10, deterministic=False)
            print(f"x_pos={x}: Stochastic mean reward: {mean_reward}")
            x_pos_rewards[x] = mean_reward

        # Update proportions to focus on poorly succeeded tasks (invert softmax)
        rewards_arr = np.array([x_pos_rewards[x] for x in x_offsets])
        sorted_indices = np.argsort(rewards_arr)  # ascending: hardest first
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(rewards_arr))
        inv_ranks = len(rewards_arr) - ranks
        proportions = inv_ranks / np.sum(inv_ranks)
        print("Updated sampling proportions for each x_pos (focus on hard tasks):", dict(zip(x_offsets, proportions)))

        # Re-create training envs with new proportions
        train_envs.close()
        env_fns = [make_weighted_env(proportions, env_params, seed + i*1000 + iteration*10000) for i in range(n_envs)]
        train_envs = SubprocVecEnv(env_fns)
        train_envs = VecNormalize(train_envs, norm_obs=False, norm_reward=True, clip_reward=100.0)
        model.set_env(train_envs)

        # Save model at each iteration
        model.save(f"ppo_checkpoints_copy2/ppo_stochastic_env_iter{iteration+1}.zip")

    # Save final model
    model.save(config.get('output_model_path', 'ppo_checkpoints_copy2/ppo_stochastic_env.zip'))
    train_envs.save("logs_copy2/vecnormalize.pkl")

if __name__ == "__main__":
    main()