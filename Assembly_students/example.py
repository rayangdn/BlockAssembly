import matplotlib.pyplot as plt
from rendering import plot_assembly_env, plot_task
#from tree import  ExtendedTree, Action
from tasks import Empty, Bridge, Tower, DoubleBridge
from assembly_env import AssemblyEnv
from blocks import Floor

# task = Empty(shapes=[Floor(xlim=(-5, 5))])
# task = Bridge(num_stories=2)
task = Tower(targets=[(0,2), (0,4), (0,6), (0,8)], obstacles=[(0,1), (0,3), (0,5), (0,7)], floor_positions=[-4, -2, 0, 2, 4])
# task = DoubleBridge(num_stories=2, with_top=True, floor_positions=[-4, -2, 0, 2, 4])

env = AssemblyEnv(task)
done = False
rewards = 0

while not done:
    action = env.random_action(non_colliding=True, stable=True)
    if action is None:
        break
    print(action)
    obs, r, done = env.step(action)
    rewards += r
    print("env is stable: ", env.is_stable())
    
print(rewards)

plot_assembly_env(env, task=task)
plt.axis('equal')
plt.show()