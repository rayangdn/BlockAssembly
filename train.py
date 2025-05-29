import os
import sys
import yaml
import numpy as np
import torch
import random
from stable_baselines3 import DQN, PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from assembly_gym_env import AssemblyGymEnv
from callbacks import InfoCallback, ReinforceEvalCallback
from reinforce import ReinforceAgent, get_next_run_number, reinforce_evaluate_policy


def mask_fn(env):
    return env.get_action_masks()

def create_task(config):
    
    sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
    from blocks import Cube, Trapezoid
    from tasks import Empty, Tower, Bridge, DoubleBridge
    
    task_type = config['task_type']
    
    # Get floor positions
    floor_positions = config['floor_positions']
    
    if task_type == 'Empty':
        return Empty(shapes=[])
    
    elif task_type == 'Tower':
        # Parse shape names into actual shape objects
        shapes = []
        for shape_name in config['tower']['shapes']:
            if shape_name == 'Cube':
                shapes.append(Cube())
            elif shape_name == 'Trapezoid':
                shapes.append(Trapezoid())
                
        return Tower(
            targets=config['tower']['targets'],
            obstacles=config['tower'].get('obstacles'),
            name=config['tower'].get('name', 'Tower'),
            floor_positions=floor_positions,
            shapes=shapes
        )
    
    elif task_type == 'Bridge':
        # Parse shape names into actual shape objects
        shapes = []
        for shape_name in config['bridge'].get('shapes', ['Cube', 'Trapezoid']):
            if shape_name == 'Cube':
                shapes.append(Cube())
            elif shape_name == 'Trapezoid':
                shapes.append(Trapezoid())
                
        return Bridge(
            num_stories=config['bridge']['num_stories'],
            width=config['bridge'].get('width', 1),
            floor_positions=floor_positions,
            shapes=shapes,
            name=config['bridge'].get('name', 'Bridge')
        )
    
    elif task_type == 'DoubleBridge':
        # Parse shape names into actual shape objects
        shapes = []
        for shape_name in config['double_bridge'].get('shapes', ['Trapezoid', 'Cube']):
            if shape_name == 'Cube':
                shapes.append(Cube())
            elif shape_name == 'Trapezoid':
                shapes.append(Trapezoid())
                
        return DoubleBridge(
            num_stories=config['double_bridge']['num_stories'],
            with_top=config['double_bridge'].get('with_top', False),
            shapes=shapes,
            floor_positions=floor_positions,
            name=config['double_bridge'].get('name', 'DoubleBridge')
        )
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
      
def make_env(config, seed=None):
    
    task = create_task(config['task'])
    
    # Extract environment configuration
    env_config = config['env']
    
    # Create the environment
    env = AssemblyGymEnv(
            task=task,  
            max_blocks=env_config['max_blocks'],
            xlim=env_config['xlim'],
            zlim=env_config['zlim'],
            img_size=tuple(env_config['img_size']),
            mu=env_config['mu'],
            density=env_config['density'],
            invalid_action_penalty=env_config['invalid_action_penalty'],
            failed_placement_penalty=env_config['failed_placement_penalty'],
            truncated_penalty=env_config['truncated_penalty'],
            max_steps=env_config['max_steps'],
            state_representation=env_config['state_representation'],
            reward_representation=env_config['reward_representation'],
        )
    
    # Visualize the environment
    # env.render(mode='human')
    
    # Set seed if provided
    if seed is not None:
        env.seed(seed)
    
    # Wrap with action masking if enabled
    if env_config['use_action_masking']:
        env = ActionMasker(env, mask_fn)
        
    return env

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train an agent on AssemblyGymEnv")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    # Set random seed for reproducibility
    seed = 42
    set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Get agent type from config
    agent_type = config['agent']['use_agent']
    
    # Create log directory
    log_dir = f"logs/{agent_type}"
    model_dir = os.path.join(log_dir, "models")
    tensorboard_log = os.path.join(log_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    # Create the environment
    env = make_env(config, seed)
    eval_env = make_env(config)
    
    # Get agent from config
    agent_config = config['agent'][agent_type].copy()
    
    # Extract policy kwargs if not using REINFORCE
    if agent_type != 'reinforce_masking':
        env = Monitor(env, log_dir)
        eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
        policy_kwargs = agent_config.pop('policy_kwargs')
        policy = agent_config.pop('policy')
    
    # Total timesteps for training from config
    total_timesteps = config['agent']['total_timesteps']
    
    # Create the agent based on type
    if agent_type == 'dqn':
        print("Using DQN agent")
        model = DQN(
            policy,
            env,
            tensorboard_log=tensorboard_log,
            verbose=config['agent']['verbose'],
            policy_kwargs=policy_kwargs,
            **agent_config
        )
        
        # Set up evaluation callback for DQN
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval_results"),
            eval_freq=5000,
            n_eval_episodes=3,
            deterministic=True,
            render=False
        )
    elif agent_type == 'ppo':
        print("Using PPO agent")
        model = PPO(
            policy,
            env,
            tensorboard_log=tensorboard_log,
            verbose=config['agent']['verbose'],
            policy_kwargs=policy_kwargs,
            **agent_config
        )
        
        # Set up evaluation callback for DQN
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval_results"),
            eval_freq=5000,
            n_eval_episodes=3,
            deterministic=True,
            render=False
        )
    elif agent_type == 'ppo_masking':
        print("Using Maskable PPO agent")
        # Make sure environment has action masking enabled
        if not config['env']['use_action_masking']:
            print("Warning: Using ppo_masking but action masking is not enabled in environment config.")
            print("Enabling action masking automatically.")
            # Recreate environments with masking
            env = ActionMasker(env.unwrapped, mask_fn)
            env = Monitor(env, log_dir)
            eval_env = ActionMasker(eval_env.unwrapped, mask_fn)
            eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
        
        model = MaskablePPO(
            policy,
            env,
            tensorboard_log=tensorboard_log,
            verbose=config['agent']['verbose'],
            policy_kwargs=policy_kwargs,
            **agent_config
        )
        
        # Set up maskable evaluation callback for MaskablePPO
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval_results"),
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
    elif agent_type == 'reinforce_masking':
        print("Using Maskable REINFORCE agent")
        # Make sure environment has action masking enabled
        if not config['env']['use_action_masking']:
            print("Warning: Using ppo_masking but action masking is not enabled in environment config.")
            print("Enabling action masking automatically.")
            # Recreate environments with masking
            env = ActionMasker(env, mask_fn)
            eval_env = ActionMasker(eval_env, mask_fn)
            
        agent_dir_name = "REINFORCE"
        run_number = get_next_run_number(tensorboard_log, agent_dir_name)
        run_name = f"{agent_dir_name}_{run_number}"
        agent_tensorboard_log = os.path.join(tensorboard_log, run_name)
        
        model = ReinforceAgent(
            env,
            learning_rate=agent_config['learning_rate'],
            gamma=agent_config['gamma'],
            tensorboard_log=agent_tensorboard_log,
            verbose=config['agent']['verbose'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create evaluation callback for REINFORCE
        eval_callback = ReinforceEvalCallback(
            eval_env=eval_env,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval_results"),
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Set up checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10000 timesteps
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix=f"{agent_type}"
    )

    # Set up the info callback
    info_callback = InfoCallback(agent_type=agent_type)
    
    # Combine all callbacks
    callbacks = CallbackList([checkpoint_callback, eval_callback, info_callback])
    
    print(f"Starting {agent_type.upper()} training for {total_timesteps} timesteps...")
    
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
    # Use the correct evaluation function based on agent type
    if agent_type == 'ppo_masking':
        mean_reward, std_reward = maskable_evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=3,
            deterministic=True
        )
    elif agent_type == 'ppo' or agent_type == 'dqn':
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=3,
            deterministic=True
        )
    elif agent_type == 'reinforce_masking':
        mean_reward, std_reward = reinforce_evaluate_policy(
            model, 
            eval_env,
            n_eval_episodes=3,
            deterministic=True
        )
    else:
        mean_reward = None
        std_reward = None
        
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
if __name__ == "__main__":
    main()