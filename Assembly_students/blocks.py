import os
import tempfile

import compas
import numpy as np
from compas.datastructures import Mesh
from compas.geometry import Frame, Scale, Translation
from compas_assembly.datastructures import Block as CRA_Block
from shapely.geometry import Point as ShapleyPoint
from shapely.geometry import Polygon as SPolygon

from geometry import merge_coplanar_faces


def attr_property(attr_name, default=None):
    """
    Helper function to create a property that gets and sets an attribute of the object.
    """
    def getter(self):
        return self.attributes.get(attr_name, default)

    def setter(self, value):
        self.attributes[attr_name] = value

    return property(getter, setter)


class Block(CRA_Block):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),  '../../'))

    @classmethod
    def from_json(cls, filepath):
        if not os.path.exists(filepath):
            filepath = os.path.join(cls.base_path, filepath)
        return super().from_json(filepath)
    
    @classmethod
    def from_mesh(cls, mesh, name=None):
        mesh = mesh.copy(cls=cls)
        if name is not None:
            mesh.name = name

        return mesh
    
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.default_face_attributes.update({
            "is_2d": False, 
            "is_receiving_2d": True, 
            "is_attaching_2d": True, 
            "offsets_x" : None,
        })

        self.default_vertex_attributes.update({"is_2d": False})

    object_id = attr_property("object_id")
    block_id = attr_property("block_id", None)
    block_id = attr_property("registered", False)
    position = attr_property("position", None)
    orientation = attr_property("orientation", None)
    fixed = attr_property("fixed", False)
    frozen = attr_property("frozen", False)

    # compare to other block
    def __eq__(self, other):
        return self.block_id == other.block_id and self.position == other.position and self.orientation == other.orientation and self.fixed == other.fixed
    
    def __hash__(self):
        position = tuple(self.position) if self.position is not None else None
        orientation = tuple(self.orientation) if self.orientation is not None else None
        return hash((self.block_id, position, orientation, self.fixed))

    def vertices_2d(self, data=False):
        if data:
            for key, attrs in self.vertices(data=True):
                if attrs['is_2d']:
                    yield key, attrs
        else:
            for vertex in self.vertices(data=False):
                if self.vertex_attribute(vertex, 'is_2d'):
                    yield vertex

    def faces_2d(self, data=False):
        for key, attrs in self.faces(data=True):
            if attrs['is_2d']:
                if data:
                    yield key, attrs
                else:
                    yield key

    def face_length_2d(self, face):
        vertices = self.face_vertices_2d(face)
        assert len(vertices) == 2
        v1, v2 = vertices
        v1, v2 = self.vertex_coordinates(v1), self.vertex_coordinates(v2)
        return np.linalg.norm(np.array(v1) - np.array(v2))
    
    def receiving_faces_2d(self, max_angle_deg=None):
        for face in self.faces_2d():
            if self.face_attribute(face, 'is_receiving_2d'):

                # check angle
                if max_angle_deg is not None:
                    face_frame = self.face_frame_2d(face)
                    # skip faces where the angle w.r.t. to the horizontal plane is too large
                    angle = np.arccos(face_frame.normal[2])
                    if angle > np.deg2rad(max_angle_deg):
                        continue

                yield face

    def attaching_faces_2d(self):
        for face in self.faces_2d():
            if self.face_attribute(face, 'is_attaching_2d'):
                yield face

    def face_vertices_2d(self, face):
        return [vertex for vertex in self.face_vertices(face) if self.vertex_attribute(vertex, 'is_2d')]

    def face_frame_2d(self, face):
        assert self.face_attribute(face, 'is_2d')

        # construct a frame for the face where the first coordinate corresponds to the x-axis
        normal = self.face_normal(face)
        yaxis = [0, 1, 0]
        return Frame(point=self.face_center(face),
                     xaxis=-np.cross(normal, yaxis),
                     yaxis=yaxis)

    def contains_2d(self, point, tol=1e-6):
        if len(point) == 3:
            point = (point[0], point[2])
        return self.polygon_2d().buffer(tol).contains(ShapleyPoint(point))
    
    def inside_box_2d(self, xlim, zlim, tol=1e-6):
        return all(
            xlim[0] - tol <= self.vertex_coordinates(v)[0] <= xlim[1] + tol and zlim[0] - tol <= self.vertex_coordinates(v)[2] <= zlim[1] + tol
            for v in self.vertices_2d()
        )
    
    def contains_2d_convex(self, points, tol=1e-6):
        contains = np.ones(len(points), dtype=bool)

        # points = np.array(points)[:, [0,2]]
        points = np.array(points)
        if points.shape[1] == 3:
            points = points[:, [0,2]]
        for i in self.faces_2d():
            frame = self.face_frame_2d(i)

            offset = np.array([frame.point[0], frame.point[2]])
            normal = np.array([frame.normal[0], frame.normal[2]])

            contains = contains & (np.dot(points - offset, normal) <= -tol)

        return contains
    
    def intersects_2d(self, other, tol=1e-6):
        if self.polygon_2d().intersects(other.polygon_2d()):
            intersection = self.polygon_2d().intersection(other.polygon_2d())
            return intersection.area >= tol
        
        return False
    
    def copy(self, cls=None, frame=None):
        block = super().copy(cls=cls)   
        if frame is not None:
            block.frame = frame
        return block

    def polygon_2d(self):
        return SPolygon([(self.vertex_coordinates(v)[0], self.vertex_coordinates(v)[2]) for v in self.vertices_2d()])
    
    def __repr__(self):
        return f"Block {self.name} ({self.node})"
    
    @property
    def __dtype__(self):
        # note: the base impementation somehow does not include the full module path
        return "{}/{}".format(".".join(self.__class__.__module__.split(".")), self.__class__.__name__)



class TemporaryURDF:

    def __init__(self, block, center=True):
        self.block = block
        self.mesh = Mesh.from_vertices_and_faces(*block.to_vertices_and_faces(triangulated=True))
        self.position = [0., 0., 0.]
        self.orientation = (0., 0., 0., 1.)
        if center:
            self.position = self.mesh.centroid()
            self.mesh.transform(Translation.from_vector(-1 * np.array(self.position)))
        self.urdf = self._create_urdf(block.name)
        self._tmpdir = None

    def _create_urdf(self, name):
        mesh_filename = f"{name}.stl"

        # Step 3: URDF template with placeholders
        urdf_template = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link concave="no" name="{name}_base_link">
    <contact>
      <lateral_friction value="0.7"/>
      <spinning_friction value=".001"/>
    </contact>
    <visual>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="package://blocks/{mesh_filename}"/>
      </geometry>
      <material name="blockmat">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="package://blocks/{mesh_filename}"/>
      </geometry>
    </collision>
    <inertial>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <mass value="1"/>
        <inertia ixx="0.1" ixy="0." ixz="0." iyy="0.1" iyz="0." izz="0.1"/>
    </inertial>
  </link>
</robot>"""
    
        return urdf_template

    def __enter__(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        # save urdf
        urdf_path = os.path.join(self._tmpdir.name, os.path.basename(self.block.name) + ".urdf") 
        with open(urdf_path, 'w') as f:
            f.write(self.urdf)
        
        # make package dir
        os.makedirs(os.path.join(self._tmpdir.name, 'blocks'))

        # save mesh
        self.mesh.to_stl(os.path.join(self._tmpdir.name, 'blocks', f"{self.block.name}.stl"), binary=True)

        return urdf_path, self.position, self.orientation

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tmpdir.cleanup()
        self._tmpdir = None


def configure_2d_faces(block):
    # detect 2d faces and vertices
    for v in block.vertices():
        if block.vertex_coordinates(v)[1] > 0:
            block.vertex_attributes(v)["is_2d"] = True

    for f in block.faces():
        if np.abs(block.face_normal(f)[1]) < 0.001:
            block.face_attributes(f)["is_2d"] = True


def reorder_faces(block):
    # reorganize faces
    new_order = [*block.faces_2d()]

    for f in block.faces():
        if f not in new_order:
            new_order.append(f)

    attributes = []
    for f in new_order:
        attributes.append( (block.face[f] , block.facedata[f]))
    
    for f in new_order:
        block.delete_face(f)

    for i, (v, a) in enumerate(attributes):
        block.add_face(v, attr_dict=a, fkey=i)


def mesh_from_2d_points(points, depth=1., center=True):
    vertices = [ np.array([points[i][0], depth, points[i][1]]) for i in range(len(points)) ] + [ np.array([points[i][0], -depth, points[i][1]]) for i in range(len(points)) ]
    faces = compas.geometry.convex_hull(vertices)
    mesh = Mesh.from_vertices_and_faces(vertices, faces)
    if center:
        mesh.transform(Translation.from_vector(-np.array(mesh.centroid())))
    return mesh


def create_block_from_points(points, name, receiving_faces=None, attaching_faces=None, center=True, offsets_x=None):
    mesh = mesh_from_2d_points(points, center=center)

    block = Block.from_mesh(mesh, name=name)
    merge_coplanar_faces(block)
    configure_2d_faces(block)
    reorder_faces(block)
    assign_block_id(block)

    if receiving_faces is not None:
        for f in block.faces_2d():
            block.face_attribute(f, "is_receiving_2d", f in receiving_faces)

    if attaching_faces is not None:
        for f in block.faces_2d():
            block.face_attribute(f, "is_attaching_2d", f in attaching_faces)

    if offsets_x is not None:
        block.default_face_attributes["offsets_x"] = offsets_x

    return block



def Cube(width=1, height=1, depth=1, receiving_faces=(1,2,3), attaching_faces=(0, ), offsets_x=None, location=None, scale=None):
    points = [(0,height), (width, height), ( width, 0), (0,0)]
    block = create_block_from_points(points, f"Cube({width}, {height})", receiving_faces=receiving_faces, attaching_faces=attaching_faces, offsets_x=offsets_x)
    if scale is not None:
        block.transform(Scale.from_factors([scale] * 3))
    if location is not None:
        block.transform(Translation.from_vector(location))
    return block


def Hexagon(attaching_faces=(0, ), receiving_faces=(1,2,3,4,5)):
    alpha = np.pi/6
    height = np.cos(alpha)
    width = np.sin(alpha)
    points = [(-1/2, -height), (-1/2 - width, 0), (-1/2, height), (1/2, height), (1/2 + width, 0), (1/2, -height)]

    return create_block_from_points(points, "Hexagon", receiving_faces=receiving_faces, attaching_faces=attaching_faces)


def Floor(xlim=(-5,5), thickness=0.05, offsets_x=None):
    width = xlim[1] - xlim[0]
    thickness = width * thickness
    points = [(xlim[0], 0), (xlim[1], 0), (xlim[1], -thickness), (xlim[0], -thickness)]

    block = create_block_from_points(points, "Floor", center=False)

    for f in block.faces_2d():
        block.face_attribute(f, "is_2d", False)
    block.face_attribute(3, "is_2d", True)
    block.face_attribute(3, "is_receiving_2d", True)
    block.face_attribute(3, "is_attaching_2d", False)
    if offsets_x is not None:
        block.face_attribute(3, "offsets_x", offsets_x)
    
    reorder_faces(block)
    
    block.fixed = True
    block.node = -1
    
    return block


def Trapezoid(index=3, stretch=1.0):
    alpha = np.pi/index/2
    height = np.cos(alpha)
    width = np.sin(alpha)
    points = [ (-stretch/2, height), (stretch/2, height), (stretch/2 + width, 0), (-stretch/2 - width, 0)]
    return create_block_from_points(points, f"Trapezoid({index}, {stretch})")


def Wedge(index=3):
    alpha = np.pi/index/2
    height = np.cos(alpha)
    width = np.sin(alpha)
    points = [ (0., height), (width, 0), ( - width, 0)]
    return create_block_from_points(points, f"Wedge({index})")


# We want unique and permanent ids for each block, to identifty actions across tasks and training runs.
# Blocks are created and registered here once and then frozen.
# New blocks cannot be registered during runtime.

BLOCKS = []
BLOCK_NAMES = []
BLOCKS_FROZEN = False

def assign_block_id(block):
    global BLOCKS
    global BLOCK_NAMES
    global BLOCKS_FROZEN

    if block.name not in BLOCK_NAMES:
        if BLOCKS_FROZEN:
            print("Warning: Block ids are frozen. No new blocks can be registered at runtime.")
            print(f"Add the block \"{block.name}\" in the file envs/blocks.py to register it.")
            return
        block.block_id = len(BLOCK_NAMES)
        BLOCK_NAMES.append(block.name)
        BLOCKS.append(block)
        block.registered = True
    
    else:
        block.registered = True
        block.block_id = BLOCK_NAMES.index(block.name)

def block_from_id(block_id):
    return BLOCKS[block_id]

Floor()
Cube()
Cube(2,1)
Hexagon()

for i in range(2,13):
    Trapezoid(i, 1.)

for i in range(2,13):
    Trapezoid(i, 0.5)

for i in range(3,9):
    Wedge(i)

BLOCKS_FROZEN = True
