import matplotlib.pyplot as plt
from rendering import plot_assembly_env, plot_task
#from tree import  ExtendedTree, Action
from tasks import Bridge
from assembly_env import AssemblyEnv
from blocks import Floor


task = Bridge(num_stories=2)

env = AssemblyEnv(task)
done = False
rewards = 0

plot_assembly_env(env, task=task, face_numbers=True)


while not done:
    action = env.random_action(non_colliding=True, stable=False)
    print(action)
    if action is None:
        break
    print(action)
    obs, r, done = env.step(action)
    rewards += r
    print(env.is_stable())
    
print(rewards)

plot_assembly_env(env, task=task, face_numbers=True)
plt.axis('equal')
plt.show()