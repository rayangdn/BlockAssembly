import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import configure

class MaskedPolicy(nn.Module):
    def __init__(self, input_shape, action_size, hidden_size=64, device='cpu'):
        super(MaskedPolicy, self).__init__()
        
        self.device = device
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = 16 * h * w  # after flattening
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        x = self.conv(state)
        logits = self.fc(x)
        return logits

    def act(self, state, action_mask):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits = self.forward(state)
        
        # Apply action mask: masked logits → -inf for invalid actions
        mask = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        logits[~mask] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

class ReinforceAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, 
                 tensorboard_log=None, verbose=0, device='cpu'):
            
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.action_size = env.action_space.n
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.device = device
        
        # Create policy and optimizer
        self.policy = MaskedPolicy(
            input_shape=self.obs_shape, 
            action_size=self.action_size,
            device=self.device
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Setup tensorboard writer
        self.writer = SummaryWriter(log_dir=tensorboard_log)
        
        # For evaluation and saving
        self.num_timesteps = 0
        self.best_mean_reward = -np.inf
        
        self.logger = configure(tensorboard_log, ["stdout", "tensorboard"])
        
    def _compute_returns(self, rewards, gamma):

        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        return returns
    
    def _compute_loss(self, log_probs, returns):

        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        return torch.stack(policy_loss).sum()
        
    def learn(self, total_timesteps, callback=None, log_interval=100):
        
        episodes = 0
        timesteps_so_far = 0
        scores_deque = deque(maxlen=100)  # last 100 scores
        
        # Initialize callback
        if callback:
            callback.init_callback(self)
            
        while timesteps_so_far < total_timesteps:
            log_probs = []
            rewards = []
            infos = []
            
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            
             # Run one episode
            while not done and episode_steps < self.env.unwrapped.max_steps:
            
                # Get action mask
                action_mask = self.env.envs[0].action_masks() if hasattr(self.env, "envs") else self.env.action_masks()
                
                # Select action
                action, log_prob = self.policy.act(obs, action_mask)
                
                # Take step in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store data
                log_probs.append(log_prob)
                rewards.append(reward)
                infos.append(info)
                
                # Update current state
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                timesteps_so_far += 1
                self.num_timesteps = timesteps_so_far
                
                # Update callback
                if callback:
                    callback.update_locals(locals())
                    if not callback.on_step():
                        break
                    
            # End of episode - compute returns and policy loss
            returns = self._compute_returns(rewards, self.gamma)
            policy_loss = self._compute_loss(log_probs, returns)
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            # Logging
            episodes += 1
            scores_deque.append(episode_reward)
            mean_score = np.mean(scores_deque)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar("rollout/ep_rew_mean", mean_score, timesteps_so_far)
                self.writer.add_scalar("rollout/ep_len_mean", episode_steps, timesteps_so_far)
                self.writer.add_scalar("train/loss", policy_loss.item(), timesteps_so_far)
            
            # Print training info
            if self.verbose >= 1 and timesteps_so_far % 100*log_interval == 0:
                print(f"Timesteps: {timesteps_so_far}/{total_timesteps}, Mean reward: {mean_score:.2f}, Episode length: {episode_steps}\n")
                
            # Check if we should save the best model
            if callback and hasattr(callback, "callbacks"):
                for cb in callback.callbacks:
                    if hasattr(cb, "best_mean_reward") and mean_score > cb.best_mean_reward:
                        cb.best_mean_reward = mean_score
                        if hasattr(cb, "best_model_save_path") and cb.best_model_save_path is not None:
                            path = os.path.join(cb.best_model_save_path, f"best_model")
                            self.save(path)
                                
        # Clean up
        if self.writer:
            self.writer.close()
        
        return self
    
    def predict(self, observation, deterministic=True):

        # Get action mask
        action_mask = self.env.envs[0].action_masks() if hasattr(self.env, "envs") else self.env.action_masks()
        
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(obs_tensor)
            
            # Apply mask
            mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
            masked_logits = logits.clone()
            masked_logits[0, ~mask] = -float('inf')
            
            if deterministic:
                # Choose action with highest probability
                action = masked_logits.argmax(dim=-1).item()
            else:
                # Sample from distribution
                probs = F.softmax(masked_logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample().item()
        
        return action, {}  # Return action and empty state dict for compatibility
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
def get_next_run_number(base_dir, agent_type):

    if not os.path.exists(base_dir):
        return 1
        
    # Get all directories that match the pattern {agent_type}_N
    existing_runs = []
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and item.startswith(f"{agent_type}_"):
            try:
                run_number = int(item.split("_")[1])
                existing_runs.append(run_number)
            except (ValueError, IndexError):
                continue
    
    # If no existing runs, return 1, otherwise return max+1
    if not existing_runs:
        return 1
    else:
        return max(existing_runs) + 1
    
def reinforce_evaluate_policy(model, env, n_eval_episodes=10, deterministic=True):
    episode_rewards = []
    episode_lengths = []
    
    for i in range(n_eval_episodes):
        # Reset the environment
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        # Run one episode
        while not done:
            
            # Get the action from the model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Execute the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update episode rewards and length
            episode_reward += reward
            episode_length += 1
                
        # Append episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward