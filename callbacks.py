import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class InfoCallback(BaseCallback):

    def __init__(self, agent_type, verbose=0):
        super(InfoCallback, self).__init__(verbose)
        # Initialize counters for cumulative tracking
        self.cumulative_invalid_actions = 0
        self.cumulative_failed_placements = 0
        self.total_steps = 0
        self.total_available_steps = 0 
        self.agent_type = agent_type
        
    def _on_step(self):

        if self.agent_type == 'reinforce_masking':
            if hasattr(self, "locals") and "info" in self.locals:
                info = self.locals["info"]
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
                    if info["is_invalid_action"]:  
                        self.cumulative_invalid_actions += 1
                    else:
                        self.total_available_steps += 1
                    self.logger.record("env/cumulative_invalid_actions", self.cumulative_invalid_actions)
                    # Add rate of invalid actions (percentage)
                    if self.total_steps > 0:
                        invalid_rate = (self.cumulative_invalid_actions / self.total_steps) * 100
                        self.logger.record("env/invalid_action_rate", invalid_rate)
                    
                # Log only cumulative failed placements (no per-step)
                if "is_failed_placement" in info:
                    if info["is_failed_placement"]:
                        self.cumulative_failed_placements += 1
                    self.logger.record("env/cumulative_failed_placements", self.cumulative_failed_placements)
                    # Add rate of failed placements (percentage)
                    if self.total_available_steps > 0:
                        failure_rate = (self.cumulative_failed_placements / self.total_available_steps) * 100
                        self.logger.record("env/failed_placement_rate", failure_rate)
        else:
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
                if info["is_invalid_action"]:  
                    self.cumulative_invalid_actions += 1
                else:
                    self.total_available_steps += 1
                self.logger.record("env/cumulative_invalid_actions", self.cumulative_invalid_actions)
                # Add rate of invalid actions (percentage)
                if self.total_steps > 0:
                    invalid_rate = (self.cumulative_invalid_actions / self.total_steps) * 100
                    self.logger.record("env/invalid_action_rate", invalid_rate)
                
            # Log only cumulative failed placements (no per-step)
            if "is_failed_placement" in info:
                if info["is_failed_placement"]:
                    self.cumulative_failed_placements += 1
                self.logger.record("env/cumulative_failed_placements", self.cumulative_failed_placements)
                # Add rate of failed placements (percentage)
                if self.total_available_steps > 0:
                    failure_rate = (self.cumulative_failed_placements / self.total_available_steps) * 100
                    self.logger.record("env/failed_placement_rate", failure_rate)
        return True
        
class ReinforceEvalCallback(BaseCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq=10000, n_eval_episodes=5, deterministic=True):
        super(ReinforceEvalCallback, self).__init__()
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        # Evaluate agent periodically
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation
            episode_rewards = []
            episode_lengths = []
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            # Log to tensorboard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            
            # Print evaluation results
            if self.verbose >= 1:
                print(f"Eval: mean reward: {mean_reward:.2f} +/- {std_reward:.2f}, mean ep length: {mean_length:.2f}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.best_model_save_path is not None:
                    os.makedirs(self.best_model_save_path, exist_ok=True)
                    self.model.save(os.path.join(self.best_model_save_path, "reinforce_best_model"))
                    if self.verbose >= 1:
                        print(f"New best model with mean reward {mean_reward:.2f}")
        
        return True