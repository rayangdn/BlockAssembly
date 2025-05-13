import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, observation_space, img_size=(64, 64), action_dims=(4, 4, 2, 4, 7)):
        super().__init__()
        in_channels = observation_space.shape[0]
        print(f"[CustomCNNMultiHead] in_channels: {in_channels}, img_size: {img_size}, action_dims: {action_dims}")

        # Convolutional encoder
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        h, w = img_size[0] // 8, img_size[1] // 8
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Linear(64 * h * w, 256)

        # Policy head: expand–bottleneck [256 → 128 -> ...]
        self.policy_fc1 = nn.Linear(256, 256)
        # Separate heads for each action component
        tb_dim, tf_dim, si_dim, fa_dim, off_dim = action_dims
        self.pi_tb  = nn.Linear(128, tb_dim)   # target_block logits
        self.pi_tf  = nn.Linear(128, tf_dim)   # target_face logits
        self.pi_si  = nn.Linear(128, si_dim)   # shape_idx logits
        self.pi_fa  = nn.Linear(128, fa_dim)   # face logits
        self.pi_off = nn.Linear(128, off_dim)  # offset_idx logits

        # Value head: [256 → 256 → 1]
        self.vf_fc1 = nn.Linear(256, 256)
        self.fc_vf  = nn.Linear(256, 1)

        self.features_dim = 256

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.shared_fc(x))
        return x

    def policy(self, x):
        x = self.forward(x)
        x = F.relu(self.policy_fc1(x))
        # Compute logits for each discrete action dimension
        logits_tb  = self.pi_tb(x)
        logits_tf  = self.pi_tf(x)
        logits_si  = self.pi_si(x)
        logits_fa  = self.pi_fa(x)
        logits_off = self.pi_off(x)
        return logits_tb, logits_tf, logits_si, logits_fa, logits_off

    def value(self, x):
        x = self.forward(x)
        x = F.relu(self.vf_fc1(x))
        return self.fc_vf(x)
