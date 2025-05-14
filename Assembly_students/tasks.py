from dataclasses import dataclass, field
import numpy as np
import random

from blocks import Cube, Floor, Trapezoid



# @dataclass
# class Target:
#     location: tuple
#     # fixed: bool = False
#     # reached: bool = False


# define a enum TaskType: OneArm, TwoArm, Glued

from enum import Enum


@dataclass
class Task:
    shapes : list
    # blocks : list
    obstacles : list = None
    targets : list = None
    freeze: bool = False
    glue : bool = False
    name : str = None
    # bounds : tuple = ((-5., -5., 0.), (5., 5., 10.))
    # mu : float = 0.8
    # density : float = 1.0
    floor_positions : list = None

    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = []
        if self.targets is None:
            self.targets = []
        if self.floor_positions is None:
            self.floor_positions = [*range(-4, 5)]

    
    @property
    def num_targets(self):
        return len(self.targets)
    
    @property
    def num_obstacles(self):
        return len(self.obstacles)
    
    # def get_floor_block(self):
    #     pass

    def __hash__(self) -> int:
        return hash((
            frozenset(self.shapes),
            frozenset(self.obstacles),
            frozenset(self.targets),
            self.freeze,
            self.glue,
            self.name,
            frozenset(self.floor_positions)
        ))

    
    # @property
    # def targets_remaining(self):
    #     return sum(1 for target in self.targets if not target.reached)
    
    # @property
    # def targets_reached(self):
    #     return sum(1 for target in self.targets if target.reached)


def Empty(shapes=None):
    if shapes is None:
        shapes = []
    return Task(name="Empty", shapes=shapes, targets=[])


def Tower(targets, obstacles=None, name="Tower", floor_positions=None, shapes=None):
    if floor_positions is None:
        floor_positions = [*range(-4, 5)]
    cube = Cube(receiving_faces=[2])
    targets = [(p, 0, h) for p, h in targets]
    if obstacles is not None:
        obstacles = [Cube(location=(p, 0, h), scale=0.5) for p, h in obstacles]
    if shapes is None:
        shapes = [cube]

    return Task(name=name, shapes=shapes, obstacles=obstacles, targets=targets, floor_positions=floor_positions)


def Bridge(num_stories, width=1, floor_positions=None, shapes=None, name="Bridge"):
    if floor_positions is None:
        floor_positions = [*range(-2, 3)]
    if shapes is None:
        shapes = [Cube(), Trapezoid()]
    
    H = 0.9
    targets = [(0, 0, num_stories * H + H/2)]
    obstacles = [Cube(location=(targets[0][0], 0., i*H + H/2), scale=0.5) for i in range(num_stories) ]
    for w in range(1, width):
        obstacles += [Cube(location=(targets[0][0]-w, 0., i*H + H/2), scale=0.5) for i in range(num_stories) ]
        obstacles += [Cube(location=(targets[0][0]+w, 0., i*H + H/2), scale=0.5) for i in range(num_stories) ]
    
    return Task(name=name, shapes=shapes, obstacles=obstacles, targets=targets, floor_positions=floor_positions)


def DoubleBridge(num_stories, with_top=False, shapes=None, floor_positions=None, name="DoubleBridge"):
    if floor_positions is None:
        floor_positions = [*range(-4, 5)]
    if shapes is None:
        trapezoid = Trapezoid() # Shape(urdf_file='shapes/trapezoid.urdf', name="trapezoid")
        cube = Cube() # Shape(urdf_file='shapes/cube1.urdf', name="cube", receiving_faces_2d=[1], target_faces_2d=[2])
        shapes = [trapezoid, cube]
   
    H = 0.9
    targets = [ (-1 , 0, num_stories * H + H/2), (1 , 0, num_stories * H + H/2) ]
    obstacles = []
    for t in  targets:
        obstacles += [Cube(location=(t[0], 0., i*H + H/2), scale=0.5) for i in range(num_stories) ]

    if with_top:
        targets.append((0, 0, (num_stories+2)*H + H/2))
        obstacles.append(Cube(location=(0, 0., (num_stories+1)*H + H/2), scale = 0.5))
        
    return Task(name=name, shapes=shapes, obstacles=obstacles, targets=targets, floor_positions=floor_positions)


def PyramideTopsOnly(name="PyramideTopsOnly", elevation=1.0, floor_positions=None, shapes=None):
    # Define the x positions and levels for the pyramid
    tower_xs = list(range(-3, 4))
    levels = [0.8, 2.0, 3.2, 4.4]  # Adjusted levels for the pyramid

    obstacles = []
    targets = []
    for x in tower_xs:
        max_level = 4 - abs(x)  # Center has 4, next 3, next 2, edge 1
        for i in range(max_level):
            z = levels[i] + elevation
            if i == max_level - 1:
                targets.append((x, z))
            else:
                obstacles.append((x, z))

    # Add a base obstacle below each tower
    for x in tower_xs:
        obstacles.append((x, 0.6))

    if floor_positions is None:
        floor_positions = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    if shapes is None:
        cube = Cube(receiving_faces=[2])
        shapes = [0, 1]  # or [cube] if you want to use the Cube object

    return Task(
        targets=[(p, 0, h) for p, h in targets],
        obstacles=[Cube(location=(p, 0, h), scale=0.5) for p, h in obstacles],
        name=name,
        floor_positions=floor_positions,
        shapes=shapes
    )


def DoubleBridgeStackedTest():
    """
    Task with two DoubleBridges:
    - One with 4 stories, bridges farther from center (e.g., at x=-3 and x=3)
    - One with 2 stories, bridges closer to center (e.g., at x=-1 and x=1)
    Both WITH tops, with the inner top a bit lower and the outer a bit higher.
    """
    H = 0.9
    # Farther double bridge (4 stories, with top slightly higher)
    far_targets = [(-3, 0, 4 * H + H/2), (3, 0, 4 * H + H/2)]
    far_obstacles = []
    for t in far_targets:
        far_obstacles += [Cube(location=(t[0], 0., i*H + H/2), scale=0.5) for i in range(4)]
    # Add top for farther bridge (slightly higher)
    far_top_z = (4+2)*H + H/2 + 0.15  # raise by 0.15
    far_targets.append((0, 0, far_top_z))
    far_obstacles.append(Cube(location=(0, 0., (4+1)*H + H/2 + 0.15), scale=0.5))

    # Closer double bridge (2 stories, with top slightly lower)
    close_targets = [(-1, 0, 2 * H + H/2), (1, 0, 2 * H + H/2)]
    close_obstacles = []
    for t in close_targets:
        close_obstacles += [Cube(location=(t[0], 0., i*H + H/2), scale=0.5) for i in range(2)]
    # Add top for closer bridge (slightly lower)
    close_top_z = (2+2)*H + H/2 - 0.10  # lower by 0.10
    close_targets.append((0, 0, close_top_z))
    close_obstacles.append(Cube(location=(0, 0., (2+1)*H + H/2 - 0.10), scale=0.5))

    # Combine all
    targets = far_targets + close_targets
    obstacles = far_obstacles + close_obstacles
    floor_positions = [*range(-4, 5)]
    shapes = [Trapezoid(), Cube()]

    return Task(
        name="DoubleBridgeStackedTest",
        shapes=shapes,
        obstacles=obstacles,
        targets=targets,
        floor_positions=floor_positions
    )


def StochasticBridge(num_stories=1, width=1, floor_positions=None, shapes=None, name="StochasticBridge", x_pos=None):
    """
    Stochastic bridge: targets and obstacles are stacked at a random x position from -2 to 2.
    """
    if floor_positions is None:
        floor_positions = [*range(-2, 3)]
    if shapes is None:
        shapes = [Cube(), Trapezoid()]
    H = 0.9
    if x_pos is None:
        x_pos = random.choice([-1.75, -0.825, 0.0, 0.825, 1.75])
    else:
        x_pos = float(x_pos)
    targets = [(x_pos, 0, num_stories * H + H/2)]
    obstacles = [Cube(location=(x_pos, 0., i*H + H/2), scale=0.5) for i in range(num_stories)]
    return Task(name=name, shapes=shapes, obstacles=obstacles, targets=targets, floor_positions=floor_positions)

