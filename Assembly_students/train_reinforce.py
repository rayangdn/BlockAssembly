import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from gym_env import AssemblyGymWrapper
from tasks import Bridge
from torch.utils.tensorboard import SummaryWriter

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskedPolicy(nn.Module):
    def __init__(self, input_shape, action_size, hidden_size=64):
        super(MaskedPolicy, self).__init__()
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
        x = state / 255.0  # normalize
        x = self.conv(x)
        logits = self.fc(x)
        return logits

    def act(self, state, action_mask):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) # add batch dim
        logits = self.forward(state)
        
        # Apply action mask: masked logits → -inf for invalid actions
        mask = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(device)
        logits[~mask] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

def reinforce_with_mask(env, policy, optimizer, gamma=1.0, n_episodes=1000, max_t=20, print_every=50):
    scores = []
    scores_deque = deque(maxlen=100)
    writer = SummaryWriter(log_dir="log/")  # tensorboard writer

    for e in range(1, n_episodes + 1):
        log_probs = []
        rewards = []

        obs, _ = env.reset()
        for t in range(max_t):
            action_mask = env.envs[0].action_masks() if hasattr(env, "envs") else env.action_masks()
            action, log_prob = policy.act(obs, action_mask)
            obs, reward, done, _, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                break

        # Compute reward-to-go
        discounts = [gamma**i for i in range(len(rewards) + 1)]
        rewards_to_go = [sum(discounts[j] * rewards[j + t] for j in range(len(rewards) - t))
                         for t in range(len(rewards))]

        # Compute policy loss
        policy_loss = []
        for log_prob, G in zip(log_probs, rewards_to_go):
            policy_loss.append(-log_prob * G)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        scores.append(total_reward)
        scores_deque.append(total_reward)

        # Log to TensorBoard
        writer.add_scalar("Reward/Total", total_reward, e)
        writer.add_scalar("Reward/Moving_Average", np.mean(scores_deque), e)
        writer.add_scalar("Loss/Policy", loss.item(), e)

        if e % print_every == 0:
            print(f"Episode {e}\tAverage Score: {np.mean(scores_deque):.2f}")

        if np.mean(scores_deque) >= 195.0:
            print(f"Environment solved in {e} episodes!")
            break

    writer.close()
    return scores


if __name__ == "__main__":
    task = Bridge(num_stories=2)
    env = AssemblyGymWrapper(task)
    env = ActionMasker(env, mask_fn)

    obs_shape = env.observation_space.shape # 1x64x64 (1 channel, 64x64 image)
    action_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MaskedPolicy(obs_shape, action_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    reinforce_with_mask(env, policy, optimizer)

        #     "algo": "MaskablePPO",
        # "n_steps": 1024,
        # "batch_size": 256,
        # "n_epochs": 4,
        # "lr": 1e-4,
        # "ent_coef": 1e-3,
        # "clip_range": 0.1,
        # "total_timesteps": 1_000_000,
        # "num_envs": 1,
        # "gamma": 0.99,
        # "gae_lambda": 0.95,
        # "clip_range": 0.1,
        # "ent_coef": 1e-3,