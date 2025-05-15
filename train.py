import os
import sys
import numpy as np
import torch
import random
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


from assembly_gym_env import AssemblyGymEnv
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from tasks import Bridge

class InfoCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(InfoCallback, self).__init__(verbose)
        # Initialize counters for cumulative tracking
        self.cumulative_invalid_actions = 0
        self.cumulative_failed_placements = 0
        self.total_steps = 0
        
    def _on_step(self):
        # Extract info from the most recent environment step
        info = self.training_env.get_attr("info")[0]
        self.total_steps += 1
        
        # Log targets reached
        if "targets_reached" in info:
            targets = info["targets_reached"]
            # Extract numbers from format like "1/5"
            if isinstance(targets, str) and '/' in targets:
                current, total = map(int, targets.split('/'))
                self.logger.record("env/targets_reached", current)
        
        # Log blocks placed
        if "blocks_placed" in info:
            self.logger.record("env/blocks_placed", info["blocks_placed"])
        
        # Log only cumulative invalid actions (no per-step)
        if "is_invalid_action" in info:
            if int(info["is_invalid_action"]):
                self.cumulative_invalid_actions += 1
            self.logger.record("env/cumulative_invalid_actions", self.cumulative_invalid_actions)
            # Add rate of invalid actions (percentage)
            if self.total_steps > 0:
                invalid_rate = (self.cumulative_invalid_actions / self.total_steps) * 100
                self.logger.record("env/invalid_action_rate", invalid_rate)
        
        # Log only cumulative failed placements (no per-step)
        if "is_failed_placement" in info:
            if int(info["is_failed_placement"]):
                self.cumulative_failed_placements += 1
            self.logger.record("env/cumulative_failed_placements", self.cumulative_failed_placements)
            # Add rate of failed placements (percentage)
            if self.total_steps > 0:
                failure_rate = (self.cumulative_failed_placements / self.total_steps) * 100
                self.logger.record("env/failed_placement_rate", failure_rate)
        
        return True

def mask_fn(env):
    return env.get_action_masks()
                
def make_env(seed=None):
    task = Bridge(num_stories=2)
    env = AssemblyGymEnv(
        task=task,
        max_blocks=10,
        xlim=(-5, 5),
        zlim=(0, 10),
        img_size=(64, 64),
        mu=0.8,
        density=1.0,
        invalid_action_penalty=1.0,
        failed_placement_penalty=0.0,
        truncated_penalty=1.0,
        max_steps=200
    )
    
    if seed is not None:
        env.seed(seed)
    
    # Wrap the environment with the ActionMasker
    env = ActionMasker(env, mask_fn)
    
    return env

def main():
    
    # Set random seed for reproducibility
    seed = 42
    set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create log directory
    log_dir = "logs/ppo_masking"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    tensorboard_log = os.path.join(log_dir, "tensorboard")

    # Create the environment
    env = make_env(seed)
    env = Monitor(env, log_dir)
    
    # Create an evaluation environment
    eval_env = make_env()
    eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
    
    # Configure policy network
    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]  
        ),
        normalize_images=False
    )
    
    # Create the PPO agent 
    model = MaskablePPO(
        "CnnPolicy",  
        env,
        learning_rate=3e-4,  # Learning rate
        n_steps=2048,        # Horizon (rollout) length
        batch_size=64,       # Minibatch size for updates
        n_epochs=10,         # Number of optimization epochs
        gamma=0.99,          # Same discount factor
        gae_lambda=0.95,     # GAE parameter
        clip_range=0.2,      # PPO clipping parameter
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1
    )

    # Set up checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10000 timesteps
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="ppo_masking",
    )

    # Set up evaluation callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=5000,  # Evaluate every 5000 timesteps
        n_eval_episodes=5,  # Number of episodes to evaluate
        deterministic=True,  # Use deterministic actions for evaluation
        render=False
    )

    # Set up the info callback
    info_callback = InfoCallback()
    # Combine all callbacks
    callbacks = CallbackList([checkpoint_callback, eval_callback, info_callback])
    
    # Total timesteps for training
    total_timesteps = 500000
    model.learn(
        total_timesteps=total_timesteps,  
        callback=callbacks,
        log_interval=100,
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")
    
    # Evaluate the trained model
    print("\nEvaluating the trained model...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=10, 
        deterministic=True
        )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
if __name__ == "__main__":
    main()