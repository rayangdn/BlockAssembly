from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from blocks import Cube, Floor, Trapezoid

# @dataclass
# class Target:
#     location: tuple
#     # fixed: bool = False
#     # reached: bool = False


# define a enum TaskType: OneArm, TwoArm, Glued


@dataclass
class Task:
    shapes: list
    # blocks : list
    obstacles: list = None
    targets: list = None
    freeze: bool = False
    glue: bool = False
    name: str = None
    # bounds : tuple = ((-5., -5., 0.), (5., 5., 10.))
    # mu : float = 0.8
    # density : float = 1.0
    floor_positions: list = None

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
        return hash(
            (
                frozenset(self.shapes),
                frozenset(self.obstacles),
                frozenset(self.targets),
                self.freeze,
                self.glue,
                self.name,
                frozenset(self.floor_positions),
            )
        )

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


def Tower(
    targets, obstacles=None, name="Tower", floor_positions=None, shapes=None
):
    if floor_positions is None:
        floor_positions = [*range(-4, 5)]
    cube = Cube(receiving_faces=[2])
    targets = [(p, 0, h) for p, h in targets]
    if obstacles is not None:
        obstacles = [Cube(location=(p, 0, h), scale=0.5) for p, h in obstacles]
    if shapes is None:
        shapes = [cube]

    return Task(
        name=name,
        shapes=shapes,
        obstacles=obstacles,
        targets=targets,
        floor_positions=floor_positions,
    )


def Bridge(
    num_stories, width=1, floor_positions=None, shapes=None, name="Bridge"
):
    if floor_positions is None:
        floor_positions = [*range(-2, 3)]
    if shapes is None:
        shapes = [Cube(), Trapezoid()]

    H = 0.9
    targets = [(0, 0, num_stories * H + H / 2)]
    obstacles = [
        Cube(location=(targets[0][0], 0.0, i * H + H / 2), scale=0.5)
        for i in range(num_stories)
    ]
    for w in range(1, width):
        obstacles += [
            Cube(location=(targets[0][0] - w, 0.0, i * H + H / 2), scale=0.5)
            for i in range(num_stories)
        ]
        obstacles += [
            Cube(location=(targets[0][0] + w, 0.0, i * H + H / 2), scale=0.5)
            for i in range(num_stories)
        ]

    return Task(
        name=name,
        shapes=shapes,
        obstacles=obstacles,
        targets=targets,
        floor_positions=floor_positions,
    )


def DoubleBridge(
    num_stories,
    with_top=False,
    shapes=None,
    floor_positions=None,
    name="DoubleBridge",
):
    if floor_positions is None:
        floor_positions = [*range(-4, 5)]
    if shapes is None:
        trapezoid = (
            Trapezoid()
        )  # Shape(urdf_file='shapes/trapezoid.urdf', name="trapezoid")
        cube = (
            Cube()
        )  # Shape(urdf_file='shapes/cube1.urdf', name="cube", receiving_faces_2d=[1], target_faces_2d=[2])
        shapes = [trapezoid, cube]

    H = 0.9
    targets = [
        (-1, 0, num_stories * H + H / 2),
        (1, 0, num_stories * H + H / 2),
    ]
    obstacles = []
    for t in targets:
        obstacles += [
            Cube(location=(t[0], 0.0, i * H + H / 2), scale=0.5)
            for i in range(num_stories)
        ]

    if with_top:
        targets.append((0, 0, (num_stories + 2) * H + H / 2))
        obstacles.append(
            Cube(location=(0, 0.0, (num_stories + 1) * H + H / 2), scale=0.5)
        )

    return Task(
        name=name,
        shapes=shapes,
        obstacles=obstacles,
        targets=targets,
        floor_positions=floor_positions,
    )
