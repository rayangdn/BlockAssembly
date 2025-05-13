import matplotlib.pyplot as plt
from rendering import plot_assembly_env
from tasks import DoubleBridgeStackedTest
from assembly_env_copy import AssemblyGymEnv
import os
import yaml
import numpy as np
from gym import spaces
from collections import namedtuple

Action = namedtuple("Action", ["target_block", "target_face", "shape", "face", "offset_x"])

def random_action_generator(max_blocks, xlim, valid_shapes, block_list_size):
    action_space = spaces.Dict({
        'target_block': spaces.Discrete(block_list_size),
        'target_face': spaces.Discrete(4),
        'shape': spaces.Discrete(len(valid_shapes)),
        'face': spaces.Discrete(4),
        'offset_x': spaces.Discrete(10),
    })
    sample = action_space.sample()
    shape_id = valid_shapes[sample['shape']]
    return np.array([
        sample['target_block'],
        sample['target_face'],
        shape_id,
        sample['face'],
        sample['offset_x']
    ]).astype(int)

# Load config and task/env dynamically
with open("/Users/tomstanic/Library/Mobile Documents/com~apple~CloudDocs/Udem/Info/BlockAssembly/Assembly_students/config.yaml", "r") as f:
    config = yaml.safe_load(f)
not_reached_penalty = config["env"].get("not_reached_penalty", 25)
valid_shapes = config["env"].get("valid_shapes", [0, 1])
max_blocks = config["env"].get("max_blocks", 5)
xlim = config["env"].get("xlim", [-1, 1])

# Use DoubleBridgeStackedTest as in your example
task = DoubleBridgeStackedTest()
env = AssemblyGymEnv(task=task, not_reached_penalty=not_reached_penalty)

obs, info = env.reset()
done = False
truncated = False
rewards = 0
i = 0
while not (done or truncated):
    i += 1
    action_array = random_action_generator(
        max_blocks=max_blocks,
        xlim=xlim,
        valid_shapes=valid_shapes,
        block_list_size=len(env.env.block_list)
    )
    obs, reward, done, truncated, info = env.step(action_array)
    rewards += reward
    if done or truncated:
        last_reward = reward
        break

print(f"Total reward: {rewards}")

assembly_env = env.env
plot_assembly_env(assembly_env, task=task, face_numbers=True)
plt.axis('equal')

block_map, face_map = obs
print("Obstacles canal 0:", (block_map == 0.09).sum())
print("Obstacles canal 1:", (face_map == 0.09).sum())
print("Valeurs uniques canal 0:", np.unique(block_map))
print("Valeurs uniques canal 1:", np.unique(face_map))
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Block ID map")
plt.imshow(block_map, cmap="rainbow")
plt.axis("off")
plt.subplot(1,2,2)
plt.title("Face ID map")
plt.imshow(face_map, cmap="tab10")
plt.axis("off")
plt.tight_layout()
plt.show()