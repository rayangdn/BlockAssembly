# Block Assembly Reinforcement Learning Environment

A comprehensive reinforcement learning framework for training agents to perform block assembly tasks. This environment simulates physical block stacking with stability constraints and supports multiple task types and RL algorithms.

## 🚀 Features

- **Multiple Task Types**: Tower building, Bridge construction, Double Bridge assembly
- **Physical Simulation**: Realistic block physics with stability and collision detection
- **Flexible State Representations**: Basic, intensity-based, and multi-channel observations
- **Multiple RL Algorithms**: DQN, PPO, Maskable PPO, and REINFORCE with action masking
- **Configurable Environment**: Easy customization through YAML configuration
- **Rich Visualization**: Real-time rendering of assembly progress

## 📋 Requirements

```bash
pip install gymnasium
pip install stable-baselines3[extra]
pip install sb3-contrib
pip install torch torchvision
pip install matplotlib
pip install numpy
pip install pyyaml
```

## 🏗️ Project Structure

```
assembly-rl/
├── assembly_gym_env.py    # Gymnasium wrapper for the assembly environment
├── train.py              # Training script for RL agents
├── test.py              # Testing and visualization script
├── config.yaml          # Configuration file for environment and agents
├── callbacks.py         # Custom training callbacks
├── reinforce.py         # Custom REINFORCE implementation
└── lib/                 # Core environment modules
    ├── assembly_env.py
    ├── tasks.py
    ├── blocks.py
    └── rendering.py
    └── stability.py
    └── geometry.py
```

## 🎮 Quick Start

### 1. Configure Your Experiment

Edit `config.yaml` to customize your experiment:

```yaml
# Choose your task
task:
  task_type: "Bridge"  # Options: "Empty", "Tower", "Bridge", "DoubleBridge"

# Configure environment
env:
  max_blocks: 5
  max_steps: 20
  state_representation: "intensity"  # "basic", "intensity", "multi_channels"
  use_action_masking: false

# Select RL algorithm
agent:
  use_agent: 'ppo'  # Options: "dqn", "ppo", "ppo_masking", "reinforce_masking"
  total_timesteps: 100000
```

### 2. Train an Agent

```bash
python train.py
```

This will:
- Create the specified task environment
- Initialize the chosen RL agent
- Train for the configured number of timesteps
- Save models and logs to `logs/{agent_type}/`
- Automatically evaluate and save the best performing model

### 3. Test Your Trained Agent

```bash
python test.py
```

This will:
- Load the best trained model
- Run 3 test episodes
- Display the assembly process with visualization
- Show performance metrics

## 🏗️ Task Types

### Tower
Build structures to reach specific target positions while avoiding obstacles.
```yaml
task:
  task_type: "Tower"
  tower:
    targets: [[1, 4], [0, 6], [-0.5, 2]]  # [x_position, height]
    obstacles: [[1, 2], [-1.5, 5]]        # [x_position, height]
    shapes: ["Cube"]
```

### Bridge
Construct bridges spanning gaps with configurable stories and width.
```yaml
task:
  task_type: "Bridge"
  bridge:
    num_stories: 2
    width: 1
    shapes: ["Cube", "Trapezoid"]
```

### DoubleBridge
Build complex double-bridge structures.
```yaml
task:
  task_type: "DoubleBridge"
  double_bridge:
    num_stories: 2
    with_top: false
    shapes: ["Cube", "Trapezoid"]
```

## 🤖 Supported RL Algorithms

### Deep Q-Network (DQN)
```yaml
agent:
  use_agent: 'dqn'
  dqn:
    learning_rate: 0.0001
    buffer_size: 50000
    exploration_fraction: 0.8
```

### Proximal Policy Optimization (PPO)
```yaml
agent:
  use_agent: 'ppo'
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
```

### Maskable PPO (with Action Masking)
```yaml
agent:
  use_agent: 'ppo_masking'
env:
  use_action_masking: true  # Required for maskable agents
```

### REINFORCE with Action Masking
```yaml
agent:
  use_agent: 'reinforce_masking'
  reinforce_masking:
    learning_rate: 0.0001
    gamma: 0.99
```

## 🎛️ State Representations

### Basic
Single-channel grayscale representation of the assembly state.

### Intensity
Enhanced single-channel with intensity-based encoding of block information.

### Multi-Channel
Separate channels for each block type plus additional feature channels.

```yaml
env:
  state_representation: "intensity"  # "basic", "intensity", "multi_channels"
```

## 📊 Monitoring Training

### TensorBoard Logs
```bash
tensorboard --logdir logs/
```

View training metrics including:
- Episode rewards
- Success rates
- Block placement statistics
- Policy losses (for policy gradient methods)

### Model Checkpoints
- **Best Model**: Saved automatically based on evaluation performance
- **Checkpoints**: Periodic saves every 10,000 timesteps
- **Final Model**: Saved at the end of training

## 🔧 Advanced Configuration

### Environment Physics
```yaml
env:
  mu: 0.8              # Friction coefficient
  density: 1.0         # Block density
  xlim: [-5, 5]        # World x-axis limits
  zlim: [0, 10]        # World z-axis limits (height)
```

### Reward Shaping
```yaml
env:
  invalid_action_penalty: 0.5
  failed_placement_penalty: 0.5
  truncated_penalty: 0.5
  reward_representation: "reshaped"  # "basic", "reshaped"
```

### Action Space
The environment automatically generates a discrete action space covering:
- Target block selection
- Target face selection
- New block shape selection
- New block orientation
- Placement offset along the target face
**: Enable action masking or reduce `max_blocks`


