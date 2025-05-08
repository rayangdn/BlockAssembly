import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from tasks import Bridge
from assembly_env import AssemblyEnv, Action
from rendering import plot_assembly_env
from blocks import Floor

class ActionSpace:
    def __init__(self, env, num_offsets=5, max_actions=200):
        self.env = env
        self.num_offsets = num_offsets
        self.valid_actions = []
        self.max_actions = max_actions
        self.update_valid_actions()
    
    def update_valid_actions(self):
        self.valid_actions = self.env.available_actions(num_block_offsets=self.num_offsets)
        if len(self.valid_actions) > self.max_actions:
            self.valid_actions = self.valid_actions[:self.max_actions]
        self.n = len(self.valid_actions)
        
    def get_action(self, idx):
        if idx < 0 or idx >= self.n:
            raise ValueError(f"Action index {idx} out of range [0, {self.n-1}]")
        return self.valid_actions[idx]

    def sample(self):
        if not self.valid_actions:
            return None, None
        idx = random.randrange(self.n)
        return self.valid_actions[idx], idx

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action_idx, reward, next_state, done, valid_action_count, next_valid_action_count):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action_idx, reward, next_state, done, 
                                    valid_action_count, next_valid_action_count)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action_idx, reward, next_state, done, valid_action_count, next_valid_action_count = zip(*batch)
        state = torch.FloatTensor(np.array(state))
        action_idx = torch.LongTensor(action_idx)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(done)
        
        return state, action_idx, reward, next_state, done, valid_action_count, next_valid_action_count
    
    def __len__(self):
        return len(self.buffer)

def preprocess_state(state):
    if isinstance(state, torch.Tensor):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)  # Add channel dimension
        return state
    else:
        state_tensor = torch.FloatTensor(state)
        if len(state_tensor.shape) == 2:
            state_tensor = state_tensor.unsqueeze(0)  # Add channel dimension
        return state_tensor
    
class DQN(nn.Module):
    def __init__(self, input_shape, max_actions=200):
        super(DQN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, max_actions)
        
    def forward(self, x, valid_action_count=None):
        
        if len(x.shape) == 3:  # Add batch dimension if needed
            x = x.unsqueeze(0)
            
        # Process through CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten and process through fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Only return Q-values for valid actions
        if valid_action_count is not None:
            # Zero out Q-values for invalid actions
            mask = torch.zeros_like(x)
            for i in range(x.size(0)):  # For each item in the batch
                count = valid_action_count[i] if isinstance(valid_action_count, tuple) else valid_action_count
                mask[i, :count] = 1
            
            x = x * mask
        return x
    
class DQNAgent:
    def __init__(self, state_dim, max_actions, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=10000):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize the DQN model
        self.policy_net = DQN(state_dim, max_actions).to(self.device)
        self.target_net = DQN(state_dim, max_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Discount factor
        self.gamma = gamma
        
    def select_action(self, state, action_space, training=True):
        # Calculate current epsilon
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  max(0, (self.epsilon_decay - self.steps_done)) / self.epsilon_decay
        self.steps_done += 1
        
        # Epsilon-greedy action selection
        if training and random.random() < epsilon:
            # Random action
            return action_space.sample()
        else:
            # Greedy action
            state = preprocess_state(state).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state, valid_action_count=action_space.n)
                action_idx = q_values[0, :action_space.n].argmax().item()
                return action_space.get_action(action_idx), action_idx
            
    def update_network(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None
        
        # Sample a batch from the replay buffer
        state, action_idx, reward, next_state, done, valid_action_count, next_valid_action_count = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action_idx = action_idx.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        
        # Compute Q-values for current states
        q_values = self.policy_net(state, valid_action_count=valid_action_count)

        state_action_values = q_values.gather(1, action_idx.unsqueeze(1))
        # Compute Q-values for next states
        next_state_values = torch.zeros(batch_size).to(self.device)
        with torch.no_grad():
            next_q_values = self.target_net(next_state, valid_action_count=next_valid_action_count)
            for i in range(batch_size):
                if not done[i]:
                    count = next_valid_action_count[i] 
                    next_state_values[i] = next_q_values[i, :count].max()
        
        # Compute expected Q-values
        expected_state_action_values = reward + (self.gamma * next_state_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']