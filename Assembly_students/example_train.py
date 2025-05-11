import matplotlib.pyplot as plt
from rendering import plot_assembly_env, plot_task
#from tree import  ExtendedTree, Action
from tasks import Bridge
from assembly_env_copy import AssemblyGymEnv 
from blocks import Floor # You'll need to import your Task class
import os

#generate a random action
import numpy as np
from gym import spaces
from collections import namedtuple

# If not already defined:
Action = namedtuple("Action", ["target_block", "target_face", "shape", "face", "offset_x"])



def random_action_generator(max_blocks, xlim, valid_shapes, block_list_size):
    action_space = spaces.Dict({
        'target_block': spaces.Discrete(block_list_size),  # Use block_list_size instead of max_blocks
        'target_face': spaces.Discrete(4),
        'shape': spaces.Discrete(len(valid_shapes)),
        'face': spaces.Discrete(4),
        'offset_x': spaces.Discrete(10),
    })

    sample = action_space.sample()
    shape_id = valid_shapes[sample['shape']]

    return np.array([
        sample['target_block'],  # Fixed: replaced action_dict with sample
        sample['target_face'],  # Fixed: replaced action_dict with sample
        shape_id,               # Use shape_id directly
        sample['face'],         # Fixed: replaced action_dict with sample
        sample['offset_x']      # Fixed: replaced action_dict with sample
    ]).astype(int)

# Create an instance of the task
task=Bridge(num_stories=1, width=2)

#create instance of AssemblyGymenvironment
env = AssemblyGymEnv(task=task)
done = False
truncated = False
rewards = 0
obs, info = env.reset()  # Newer gym versions return (obs, info)


valid_figs = []

while not (done or truncated):
    #action = env.random_action()
    action_array = random_action_generator(
        max_blocks=5,
        xlim=[-1, 1],
        valid_shapes=[0, 1],
        block_list_size=len(env.env.block_list)
    )


    # Pass the Action object to env.env.step
    obs, reward, done, truncated, info = env.step(action_array)
    rewards += reward
    if done or truncated:
        print(f"last{reward}")
        break


print(f"Total reward: {rewards}")



assembly_env = env.env
plot_assembly_env(assembly_env, task=task, face_numbers=True)
plt.axis('equal')

block_map, face_map = obs  # shape: (64, 64) from (1, 64, 64)


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Block ID map")
plt.imshow(block_map, cmap="tab20")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Face ID map")
plt.imshow(face_map, cmap="tab10")
plt.axis("off")

plt.tight_layout()
plt.show()

