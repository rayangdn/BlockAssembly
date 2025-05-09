import os
import sys
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from  assembly_gym_env import AssemblyGymEnv
from tasks import Bridge
from dqn_agent import AssemblyCNN


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