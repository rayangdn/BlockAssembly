import os
import sys
import numpy as np
import torch
import random
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

class InfoCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(InfoCallback, self).__init__(verbose)
        # Initialize counters for cumulative tracking
        self.cumulative_invalid_actions = 0
        self.cumulative_failed_placements = 0
        self.total_steps = 0
        
    def _on_step(self) -> bool:
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

from assembly_gym_env import AssemblyGymEnv
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from tasks import Bridge
                
def make_env():
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
        failed_placement_penalty=0.5,
        truncated_penalty=1.0,
        max_steps=200
    )
    return env

def main():
    
    # Set random seed for reproducibility
    seed = 42
    set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Create log directory
    log_dir = "logs/dqn_no_masking"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    tensorboard_log = os.path.join(log_dir, "tensorboard")

    # Create the environment
    env = make_env()
    env = Monitor(env, log_dir)
    
    # Create an evaluation environment
    eval_env = make_env()
    eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
    
    # Configure policy network
    policy_kwargs = dict(
        net_arch=[64, 64],  # Hidden layer sizes
    )
    
    # Create the DQN agent 
    model = DQN(
        "CnnPolicy",  
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,  # Target network update rate
        gamma=0.99,  # Discount factor
        train_freq=(4, "step"),  # Update the model every 4 steps
        gradient_steps=1,  # How many gradient updates per update
        target_update_interval=1000,  # Target network update frequency
        exploration_fraction=0.6,  # Fraction of training to reduce epsilon
        exploration_initial_eps=1.0,  # Initial random action probability
        exploration_final_eps=0.05,  # Final random action probability
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1
    )

    # Set up checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10000 timesteps
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="dqn_no_masking"
    )

    # Set up evaluation callback
    eval_callback = EvalCallback(
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
    
    # Run visualization test
    print("\nRunning visualization tests...")
    test_env = make_env()
    
    for episode in range(3): 
        obs, _ = test_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            step_count += 1
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
        print(f"Targets reached: {info['targets_reached']}, Blocks placed: {info['blocks_placed']}")
        test_env.render()
    
if __name__ == "__main__":
    main()