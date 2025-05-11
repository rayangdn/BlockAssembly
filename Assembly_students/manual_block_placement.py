import matplotlib.pyplot as plt
from rendering import plot_assembly_env
from tasks import Bridge
from assembly_env_copy import AssemblyGymEnv
import numpy as np

# Crée une instance de la tâche
# Tu peux modifier les paramètres ici si besoin
TASK = Bridge(num_stories=1, width=2)

# Crée l'environnement
env = AssemblyGymEnv(task=TASK)

def demander_action(block_list_size, valid_shapes):
    print("\nEntrez les paramètres pour placer un bloc :")
    target_block = int(input(f"target_block (0 à {block_list_size-1}): "))
    target_face = int(input("target_face (0 à 3): "))
    shape = int(input(f"shape (valeurs valides: {valid_shapes}): "))
    face = int(input("face (0 à 3): "))
    offset_x = int(input("offset_x (0 à 9): "))
    return np.array([target_block, target_face, shape, face, offset_x]).astype(int)

obs, info = env.reset()
done = False
truncated = False
rewards = 0

while not (done or truncated):
    action = demander_action(len(env.env.block_list), [0, 1])
    obs, reward, done, truncated, info = env.step(action)
    rewards += reward
    print(f"Récompense de l'action: {reward}")
    print(info['cause'])
    # Affiche le plot de l'état courant après chaque action
    plot_assembly_env(env.env, task=TASK, face_numbers=True)
    plt.axis('equal')
    plt.show()
    if done or truncated:
        print("\nFin de l'épisode.")
        break

print(f"\nRécompense totale: {rewards}")

assembly_env = env.env
plot_assembly_env(assembly_env, task=TASK, face_numbers=True)
plt.axis('equal')
plt.show()
