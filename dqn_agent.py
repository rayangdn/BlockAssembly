import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Custom CNN feature extractor for the assembly environment
class AssemblyCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Get the shape of the input images
        # Assuming observation is a single-channel 2D grid
        n_input_channels = 1
        
        # CNN architecture
        self.cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second convolutional layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third convolutional layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = observation_space.sample()[None].astype(np.float32)
            sample_tensor = th.as_tensor(sample).float()
            # Add channel dimension if needed 
            if len(sample_tensor.shape) == 3:  # batch, height, width
                sample_tensor = sample_tensor.unsqueeze(1)  # Add channel dimension
            n_flatten = self.cnn(sample_tensor).shape[1]
        
        # Fully connected layer to get the features_dim output
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Add channel dimension if needed (for 2D grid observations)
        if len(observations.shape) == 3:  # batch, height, width
            observations = observations.unsqueeze(1)  # Add channel dimension
        
        # Pass through CNN layers
        cnn_features = self.cnn(observations)
        
        # Pass through linear layers
        return self.linear(cnn_features)

