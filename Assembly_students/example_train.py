import matplotlib.pyplot as plt
from rendering import plot_assembly_env, plot_task
#from tree import  ExtendedTree, Action
from tasks import Bridge
from assembly_env import AssemblyGymEnv 
from blocks import Floor # You'll need to import your Task class
import os

#generate a random action
import numpy as np
from gym import spaces
from collections import namedtuple

# If not already defined:
Action = namedtuple("Action", ["target_block", "target_face", "shape", "face", "offset_x"])


def random_action_generator_float(max_blocks=5,
                                  xlim=(-4.0, 4.0),
                                  valid_shapes=(0, 1)):
    # Define every component as a continuous Box so samples are floats
    action_space = spaces.Dict({
        'target_block': spaces.Box(low=0.0,
                                   high=float(max_blocks - 1),
                                   shape=(1,),
                                   dtype=np.float32),
        'target_face': spaces.Box(low=0.0,
                                  high=3.0,
                                  shape=(1,),
                                  dtype=np.float32),
        'shape': spaces.Box(low=0.0,
                            high=float(len(valid_shapes) - 1),
                            shape=(1,),
                            dtype=np.float32),
        'face': spaces.Box(low=0.0,
                           high=3.0,
                           shape=(1,),
                           dtype=np.float32),
        'offset_x': spaces.Box(low=float(xlim[0]),
                               high=float(xlim[1]),
                               shape=(1,),
                               dtype=np.float32),
    })

    sample = action_space.sample()

    # Round & clip the shape index back to a valid integer, then map
    raw_shape_idx = int(np.clip(np.round(sample['shape'][0]), 0, len(valid_shapes)-1))
    shape_id = valid_shapes[raw_shape_idx]

    return {
        'target_block': float(sample['target_block'][0]),
        'target_face':  float(sample['target_face'][0]),
        'shape':        float(shape_id),
        'face':         float(sample['face'][0]),
        'offset_x':     float(sample['offset_x'][0]),
    }

def random_action_generator(max_blocks = 5, xlim = [-4, 4], valid_shapes=[0, 1]):
    action_space = spaces.Dict({
        'target_block': spaces.Discrete(max_blocks),
        'target_face': spaces.Discrete(4),
        'shape': spaces.Discrete(len(valid_shapes)),
        'face': spaces.Discrete(4),
        'offset_x': spaces.Box(low=float(xlim[0]), high=float(xlim[1]), shape=(1,), dtype=np.float32),
    })

    
    sample = action_space.sample()

    # Map shape index (0 or 1) to actual shape ID (e.g., 1 or 5)
    shape_id = valid_shapes[sample['shape']]

    return {
        'target_block': sample['target_block'],
        'target_face': sample['target_face'],
        'shape': shape_id,
        'face': sample['face'],
        'offset_x': float(sample['offset_x'][0])  # Extract scalar
    }

# Create an instance of the task
task=Bridge(num_stories=1)

#create instance of AssemblyGymenvironment
env = AssemblyGymEnv(task=task)
done = False
rewards = 0
obs, info = env.reset()  # Newer gym versions return (obs, info)


valid_figs = []

while not done:
    #action = env.random_action()
    action_dict = random_action_generator_float(max_blocks=5, xlim=[-5, 5], valid_shapes=[0, 1])
    print(action_dict)

    action_array = [
            action_dict['target_block'],
            action_dict['target_face'],
            action_dict['shape'],
            action_dict['face'],
            action_dict['offset_x']
        ]

    obs, reward, done, truncated, info = env.step(action_array)
    rewards += reward
    print(rewards)
    print('-----------------------')
    print(info['cause'])
    print('-----------------------')




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

