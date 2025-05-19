import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_learning_efficiency(task='bridge'):
    # Set the style for the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    
    result_dir = f"results/{task}"
    os.makedirs(result_dir, exist_ok=True)

    # Read the data files
    dqn_data = pd.read_csv(os.path.join(result_dir, "dqn_tensorboard_DQN_1.csv"))
    ppo_data = pd.read_csv(os.path.join(result_dir, "ppo_tensorboard_PPO_1.csv"))
    reinforce_data = pd.read_csv(os.path.join(result_dir, "reinforce_masking_tensorboard_REINFORCE_1.csv"))
    masked_ppo_data = pd.read_csv(os.path.join(result_dir, "ppo_masking_tensorboard_PPO_1.csv"))

    # Create figure and axis objects with a specific figure size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the data
    ax.plot(dqn_data['Step'], dqn_data['Value'], 'b', linewidth=2, markersize=8, label='DQN')
    ax.plot(ppo_data['Step'], ppo_data['Value'], 'r', linewidth=2, markersize=8, label='PPO')
    ax.plot(reinforce_data['Step'], reinforce_data['Value'], 'g', linewidth=2, markersize=8, label='Masked-REINFORCE')
    ax.plot(masked_ppo_data['Step'], masked_ppo_data['Value'], 'purple', linewidth=2, markersize=8, label='Masked-PPO')
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    # Customize the plot
    ax.set_xlabel('Training Steps', fontsize=22, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=22, fontweight='bold')
    #ax.set_title('Learning Efficiency Comparison', fontsize=18, fontweight='bold')

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Customize the tick labels
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Format x-axis to display steps in thousands
    def format_func(value, tick_number):
        return f'{int(value/1000)}k'
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # Add legend with custom styling
    legend = ax.legend(loc='best', frameon=True, fontsize=22)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('lightgray')

    # Add annotations for the maximum rewards
    max_dqn_idx = dqn_data['Value'].idxmax()
    max_dqn_step = dqn_data.loc[max_dqn_idx, 'Step']
    max_dqn_value = dqn_data.loc[max_dqn_idx, 'Value']
    
    max_ppo_idx = ppo_data['Value'].idxmax()
    max_ppo_step = ppo_data.loc[max_ppo_idx, 'Step']
    max_ppo_value = ppo_data.loc[max_ppo_idx, 'Value']
    
    max_reinforce_idx = reinforce_data['Value'].idxmax()
    max_reinforce_step = reinforce_data.loc[max_reinforce_idx, 'Step']
    max_reinforce_value = reinforce_data.loc[max_reinforce_idx, 'Value']
    
    max_masked_ppo_idx = masked_ppo_data['Value'].idxmax()
    max_masked_ppo_step = masked_ppo_data.loc[max_masked_ppo_idx, 'Step']
    max_masked_ppo_value = masked_ppo_data.loc[max_masked_ppo_idx, 'Value']

    # ax.annotate(f'Max: {max_dqn_value:.2f}', 
    #             xy=(max_dqn_step, max_dqn_value),
    #             xytext=(0, 10), textcoords='offset points',
    #             fontsize=18, color='blue', fontweight='bold')

    # ax.annotate(f'Max: {max_ppo_value:.2f}', 
    #             xy=(max_ppo_step, max_ppo_value),
    #             xytext=(-60, -30), textcoords='offset points',
    #             fontsize=18, color='red', fontweight='bold')
    
    # ax.annotate(f'Max: {max_reinforce_value:.2f}',
    #             xy=(max_reinforce_step, max_reinforce_value),
    #             xytext=(530, -30), textcoords='offset points',
    #             fontsize=18, color='green', fontweight='bold')
    
    # ax.annotate(f'Max: {max_masked_ppo_value:.2f}', 
    #         xy=(max_masked_ppo_step, max_masked_ppo_value),
    #         xytext=(-60, -30), textcoords='offset points',
    #         fontsize=18, color='purple', fontweight='bold')

    # Add a subtle background gradient
    ax.set_facecolor('#f8f9fa')

    # Add a box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('lightgray')

    # Adjust layout
    plt.tight_layout()

    # Save the figure (optional)
    plt.savefig(os.path.join(result_dir, "learning_efficiency.png"), dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
    
plot_learning_efficiency(task='bridge')
