import os
import sys
import yaml
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from typing import Dict, Any

# Import necessary modules and functions from train.py
from train import create_task, mask_fn, make_env

def main():
    
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Get agent type from config
    agent_type = config['agent']['use_agent']
    
    # Create the environment
    test_env = make_env(config)
    
    # Determine model type and path based on agent_type
    if 'ppo_masking' in agent_type:
        model_class = MaskablePPO
    elif 'ppo' in agent_type:
        model_class = PPO
    elif 'dqn' in agent_type:
        model_class = DQN
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Model path 
    best_model_path = f"./logs/{agent_type}/models/best_model/best_model.zip"
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"Loading best model from {model_path}")
    else:
        raise FileNotFoundError(f"No model found for {agent_type}. Please train a model first.")
    
    # Load the model
    model = model_class.load(model_path, env=test_env)
    
    # Run visualization test
    print("\nRunning visualization tests...")
    
    for episode in range(3):
        obs, _ = test_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        while not (done or truncated):
            
            # Get action masks for MaskablePPO if needed
            if isinstance(model, MaskablePPO):
                action_masks = test_env.get_action_masks()
                action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _states = model.predict(obs, deterministic=True)
                
            obs, reward, done, truncated, info = test_env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Optional: print step information
            # print(f"Step {step_count}: Reward = {reward:.2f}, Action = {action}")
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
        print(f"Targets reached: {info.get('targets_reached', 'N/A')}, Blocks placed: {info.get('blocks_placed', 'N/A')}")
        
        # Render the final state
        test_env.render()


if __name__ == "__main__":
    main()