import sys
import os
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from tasks import Bridge
from assembly_env import AssemblyEnv
from rendering import plot_assembly_env
from blocks import Floor
from agent import DQNAgent, ActionSpace, ReplayBuffer, preprocess_state

def train_agent(env, agent, replay_buffer, num_episodes=1000, batch_size=32,
                target_update=10, max_steps=1000, eval_interval=100, max_actions=200):
    
    episode_rewards = []
    losses = []
    
    for episode in range(num_episodes):
        env.reset()
        env.add_block(Floor(xlim=env.xlim))  # Add the floor back
        env.num_targets_reached = 0  # Reset target counter
        env.state_feature = torch.zeros(env.img_size)  # Reset state feature
        state = env.state_feature
        action_space = ActionSpace(env, num_offsets=5, max_actions=max_actions)
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select and perform an action
            action, action_idx = agent.select_action(state, action_space, training=True)
            if action is None:
                break
            
            next_state, reward, done = env.step(action)
            if next_state is None:
                next_state = state.clone()
            state = preprocess_state(state)
            next_state= preprocess_state(next_state)
            
            episode_reward += reward.item()
            
            # Update action space
            action_space.update_valid_actions()

            # Store the transition in replay buffer
            replay_buffer.push(
                state.clone().numpy(),
                action_idx,
                reward.item(),
                next_state.clone().numpy(),
                done,
                action_space.n,
                action_space.n if not done else 0
            )
            
            # Move to the next state
            state = next_state
            # Perform a training step
            loss = agent.update_network(replay_buffer, batch_size)
            if loss is not None:
                episode_loss.append(loss)
                
            if done:
                break
            
        # Update the target network
        if episode % target_update == 0:
            agent.update_target_network()
            
        # Log results
        episode_rewards.append(episode_reward)
        if episode_loss:
            losses.append(sum(episode_loss) / len(episode_loss))
        else:
            losses.append(0)
            
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, "
                  f"Avg Loss: {losses[-1]:.4f}, Epsilon: {agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * max(0, (agent.epsilon_decay - agent.steps_done)) / agent.epsilon_decay:.2f}")
            
        # Evaluate the agent
        if episode % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, num_episodes=5)
            print(f"Evaluation: Average Reward over 5 episodes: {eval_reward:.2f}")
            
    return episode_rewards, losses
            
def evaluate_agent(env, agent, num_episodes=5, max_steps=1000):

    total_reward = 0
    
    for episode in range(num_episodes):
        env.reset()
        env.add_block(Floor(xlim=env.xlim))  # Add the floor back
        env.num_targets_reached = 0  # Reset target counter
        env.state_feature = torch.zeros(env.img_size)  # Reset state feature
        state = env.state_feature
        action_space = ActionSpace(env)
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action without exploration
            action, _ = agent.select_action(state, action_space, training=False)
            if action is None:
                break
                
            state, reward, done = env.step(action)
            episode_reward += reward.item()
            
            # Update action space
            action_space.update_valid_actions()
            
            if done:
                break
                
        total_reward += episode_reward
        
    return total_reward / num_episodes
        
def main():
    
    # Create environment
    task = Bridge(num_stories=4)
    env = AssemblyEnv(task)
    
    # Set hyperparameters
    state_dim = (1, 64, 64)  # (channels, height, width)
    max_actions = 160  # Maximum number of possible actions (not to high, training will be slow)
    learning_rate = 0.001
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 10000
    batch_size = 32
    replay_buffer_size = 10000
    num_episodes = 1000
    target_update = 10
    
    # Initialize agent and replay buffer
    agent = DQNAgent(
        state_dim=state_dim,
        max_actions=max_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay
    )
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
   
    # Train the agent
    episode_rewards, losses = train_agent(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        num_episodes=num_episodes,
        batch_size=batch_size,
        target_update=target_update,
        max_actions=max_actions
    )
    
    # Save the trained agent
    agent.save("dqn_assembly_agent.pth")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig("dqn_training_results.png")
    plt.show()
    
    # Visualize a few episodes with the trained agent
    for i in range(3):
        env.reset()
        env.add_block(Floor(xlim=env.xlim))
        env.num_targets_reached = 0
        env.state_feature = torch.zeros(env.img_size)
        state = env.state_feature
        action_space = ActionSpace(env)
        done = False
        
        while not done:
            action, _ = agent.select_action(state, action_space, training=False)
            if action is None:
                break
                
            _, reward, done = env.step(action)
            action_space.update_valid_actions()
            state = env.state_feature
        
        # Plot the final assembly
        plot_assembly_env(env, task=task)
        plt.axis('equal')
        plt.title(f"Episode {i+1} - Final Assembly")
        plt.savefig(f"dqn_episode_{i+1}_final.png")
        plt.show()
    # Plot the initial state feature map
    # plt.figure(figsize=(8, 6))
    # plt.imshow(env.state_feature, cmap='viridis', origin='upper')
    # plt.colorbar(label='Feature Value')
    # plt.title('Initial State Feature Map')
    # plt.xlabel('X-axis')
    # plt.ylabel('Z-axis')
    # plt.savefig('state_feature_map.png')
    # plt.show()
    
    # # Plot the reward feature map
    # plt.figure(figsize=(8, 6))
    # plt.imshow(env.reward_feature, cmap='plasma', origin='upper')
    # plt.colorbar(label='Reward Value')
    # plt.title('Reward Feature Map')
    # plt.xlabel('X-axis')
    # plt.ylabel('Z-axis')
    # plt.show()
    
    # # Plot the assembly environment (shows blocks, obstacles, and targets)
    # fig, ax = plt.subplots(figsize=(8, 6))
    # plot_assembly_env(env, fig=fig, ax=ax, task=task)
    # plt.title('Assembly Environment')
    # plt.axis('equal')
    # plt.show()
    
if __name__ == "__main__":
    main()