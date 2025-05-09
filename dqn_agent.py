import os
import sys
import time
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces

# Import your environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from assembly_gym_env import AssemblyGymEnv
from tasks import Bridge

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

def main():
    
    
    # Create log directory
    log_dir = "dqn_assembly_logs"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    eval_dir = os.path.join(log_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Create the environment
    task = Bridge(num_stories=2)
    env = AssemblyGymEnv(task, max_blocks=10)

    # Wrap the environment with Monitor to log statistics
    env = Monitor(env, log_dir)

    # Create a separate environment for evaluation
    eval_env = AssemblyGymEnv(task, max_blocks=10)
    eval_env = Monitor(eval_env, eval_dir)

    # Policy keyword arguments for CNN feature extraction
    policy_kwargs = dict(
        features_extractor_class=AssemblyCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 128]  # Size of the hidden layers in the policy network
    )

    # Create the DQN agent 
    model = DQN(
        policy="CnnPolicy", 
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=50000,  
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.4, 
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=log_dir
    )

    # Callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="dqn_assembly"
    )

    # Evaluation callback to monitor performance on evaluation environment
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=eval_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Train the model
    start_time = time.time()

    model.learn(
        total_timesteps=50000,  
        callback=[checkpoint_callback, eval_callback]
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save the final model
    model.save(os.path.join(model_dir, "final_model"))
    
if __name__ == "__main__":
    main()