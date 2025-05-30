import matplotlib.pyplot as plt
from rendering import plot_assembly_env, plot_task
from tasks import Bridge
from assembly_env import AssemblyEnv, Action
from blocks import Floor



task = Bridge(num_stories=2)

env = AssemblyEnv(task)

action = Action(target_block=0, target_face=0, shape=1, face = 0, offset_x = -1)
obs, r, done = env.step(action)

for a in env.available_actions(num_block_offsets=5):
    new_block = env.create_block(a)
    if env.collision(new_block):
        continue
    env.add_block(new_block)  
    if env.is_stable():
        print(a)
    env.delete_block(list(env.nodes())[-1])

if False:
    action = Action(target_block=1, target_face=3, shape=5, face = 2, offset_x = -0.)
    obs, r, done = env.step(action)

    action = Action(target_block=0, target_face=0, shape=1, face = 0, offset_x = 1.5)
    obs, r, done = env.step(action)

    action = Action(target_block=3, target_face=3, shape=5, face = 1, offset_x = 0.)
    obs, r, done = env.step(action)

if False:
    action = Action(target_block=0, target_face=0, shape=1, face = 0, offset_x = -1.5)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    action = Action(target_block=1, target_face=3, shape=5, face = 2, offset_x = 0.)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    action = Action(target_block=0, target_face=0, shape=1, face = 0, offset_x = 1.5)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    action = Action(target_block=3, target_face=3, shape=5, face = 1, offset_x = 0.)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    action = Action(target_block=4, target_face=2, shape=5, face = 1, offset_x = 0.)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    print(done)


plot_assembly_env(env, task=task)
plt.axis('equal')
plt.show()