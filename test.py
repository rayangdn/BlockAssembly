import os
import sys
from stable_baselines3 import DQN  

from assembly_gym_env import AssemblyGymEnv
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from tasks import Bridge

def make_env(seed=None):
    task = Bridge(num_stories=2)
    env = AssemblyGymEnv(
        task=task,
        max_blocks=10,
        xlim=(-5, 5),
        zlim=(0, 10),
        img_size=(64, 64),
        mu=0.8,
        density=1.0,
        invalid_action_penalty=1.0,
        failed_placement_penalty=0.0,
        truncated_penalty=1.0,
        max_steps=200
    )
    
    if seed is not None:
        env.seed(seed)
    
    return env

def main():
    
    # Run visualization test
    print("\nRunning visualization tests...")
    test_env = make_env()
    
    # Load the trained model
    model = DQN.load("logs/dqn_no_masking/models/best_model/best_model.zip", env=test_env)
    
    for episode in range(3): 
        obs, _ = test_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            step_count += 1
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
        print(f"Targets reached: {info['targets_reached']}, Blocks placed: {info['blocks_placed']}")
        test_env.render()
    
if __name__ == "__main__":
    main()