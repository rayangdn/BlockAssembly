import matplotlib.pyplot as plt
from rendering import plot_assembly_env
import tasks
from tasks import DoubleBridgeStackedTest
from assembly_env_copy_2 import AssemblyGymEnv
import os
import yaml
import numpy as np
from gym import spaces
from collections import namedtuple

Action = namedtuple("Action", ["target_block", "target_face", "shape", "face", "offset_x"])

def random_action_generator(max_blocks):
    action_space = spaces.Dict({
        'target_block': spaces.Discrete(max_blocks),
        'target_face': spaces.Discrete(4),
        'shape': spaces.Discrete(2),
        'face': spaces.Discrete(4),
        'offset_x': spaces.Discrete(5),
    })
    sample = action_space.sample()
    return np.array([
        sample['target_block'],
        sample['target_face'],
        sample['shape'],
        sample['face'],
        sample['offset_x']
    ]).astype(int)

# Load config and task/env dynamically
with open("config copy 2.yaml", "r") as f:
    config = yaml.safe_load(f)
not_reached_penalty = config["env"].get("not_reached_penalty", 25)
max_blocks = config["env"].get("max_blocks", 5)
xlim = config["env"].get("xlim", [-1, 1])

# Build task from config (dynamic type and params)
#from tasks import task_fn  # Import task_fn from the appropriate module

task_cfg = config["task"]
task_type = task_cfg["type"]
task_fn = getattr(tasks, task_type)
task_args = task_cfg.get("args", {})
task = task_fn(**task_args) if task_args else task_fn()

# Prepare environment arguments from config
env_cfg = config.get("env", {})  # Load env_cfg from the configuration file
env = AssemblyGymEnv(
    task=task,
    max_blocks=env_cfg.get("max_blocks", 10),
    n_offsets=env_cfg.get("n_offsets", 10),
    limit_steps=env_cfg.get("limit_steps", 2),
    end_reward=env_cfg.get("end_reward", 100),
    n_floor=env_cfg.get("n_floor", 0),
    target_reward_per_block=env_cfg.get("target_reward_per_block", 10),
    min_block_reach_target=env_cfg.get("min_block_reach_target", 8),
    collision_penalty=env_cfg.get("collision_penalty", 0),
    unstable_penalty=env_cfg.get("unstable_penalty", 0),
    not_reached_penalty=env_cfg.get("not_reached_penalty", 0),
)




obs, info = env.reset()
done = False
truncated = False
rewards = 0
i = 0
while not (done or truncated):
    i += 1
    action_array = random_action_generator(
        max_blocks = env_cfg.get("max_blocks", 10)
    )
    obs, reward, done, truncated, info = env.step(action_array)
    rewards += reward
    if done or truncated:
        last_reward = reward
        break

print(f"Total reward: {rewards}")
print(f"steps: {i}")

# Visualization and state extraction
state_feature = env.env.state_feature.numpy()
plt.figure(figsize=(12, 4))
for idx in range(state_feature.shape[0]):
    plt.subplot(1, state_feature.shape[0], idx + 1)
    plt.title(f"Channel {idx}")
    plt.imshow(state_feature[idx], cmap="viridis", vmin=0.0, vmax=1.0)
    plt.axis("off")
plt.suptitle("All state feature channels")
plt.tight_layout()
plt.show()

plot_assembly_env(
    env,
    fig=None, ax=None,
    plot_forces=False, force_scale=1.0,
    plot_edges=False, equal=True,
    face_numbers=True, nodes=False,
    task=task
)
plt.axis('equal')
plt.show()