import math
import numpy as np
from shapely.geometry import Polygon, Point
from compas.geometry import Plane, Point, Rotation, distance_point_point, Transformation, Frame


def merge_coplanar_faces(mesh):
    faces = [*mesh.faces()]
    while len(faces) > 0:
        face = faces.pop()

        # look at neighbouring faces
        for face2 in mesh.face_neighborhood(face):
            points = mesh.face_coordinates(face) + mesh.face_coordinates(face2)
            points = [Point(*p) for p in points]
            
            # merge faces if they are coplanar
            if is_coplanar(points):
                new_face = mesh.merge_faces([face, face2])
                faces.remove(face2)
                faces.append(new_face)
                break


def contains_point(plane, point, tol=1e-6):
    # compute distance from point to plane
    return np.abs(np.inner(point - plane.point, plane.normal)) <= tol


def is_coplanar(points):
    if len(points) < 4:
        return True
    plane = Plane.from_three_points(*points[:3])
    for p in points[3:]:
        if not contains_point(plane, p, tol=1e-3):
            return False
    return True


def align_frames_2d(frame1, frame2, frame1_coordinates=None):
    """
    Compute the offset and rotation to align frame1 with frame2 in 2D, i.e. rotate around the y axis.
    """
    if frame1_coordinates is None:
        frame1_coordinates = [0, 0, 0]
    n1 = frame1.normal
    n2 = frame2.normal
    rotation = Rotation.from_axis_and_angle(np.cross(n1, n2) + np.array([0, 1e-6, 0]),
                                            np.arccos(np.clip(-np.dot(n1, n2), -1.0, 1.0))).rotation
    offset = frame1.to_world_coordinates(frame1_coordinates) - frame2.point.transformed(rotation)
    return offset, rotation


def align_blocks(block1, face1, block2, face2, frame1_coordinates=None):
    if frame1_coordinates is None:
        frame1_coordinates = [0, 0, 0]
    position, rotation = align_frames_2d(block1.face_frame_2d(face=face1), block2.face_frame_2d(face=face2), frame1_coordinates=frame1_coordinates)
    frame = Frame.from_rotation(rotation=rotation, point=position)
    new_block = block2.transformed(Transformation.from_frame(frame))
    new_block.position = position
    new_block.orientation = rotation.quaternion
    return new_block

def affine_transform_vertices(vertices, shift, rotation):
    rot = Rotation.from_euler("xyz", rotation)
    return shift + rot.apply(vertices)


def bounding_box_collision(mesh1, mesh2):
    box1 = mesh1.aabb()
    box2 = mesh2.aabb()

    for point in box1.points:
        if box2.contains_point(point):
            return True
    
    for point in box2.points:
        if box1.contains_point(point):
            return True
    
    return False


def check_collision2D(shape1, shape2):  # only works in 2D: the block must be aligned wrt to the y axis
    vertices1_2d = np.delete(shape1[np.where(shape1[:, 1] > 0)], 1, 1)
    vertices2_2d = np.delete(shape2[np.where(shape2[:, 1] > 0)], 1, 1)

    polygon1 = Polygon(vertices1_2d)
    polygon2 = Polygon(vertices2_2d)

    return polygon1.intersects(polygon2)


def quaternion_distance(q1, q2):
    """
    Computes the angle between two quaternions
    """
    return np.arccos(min(2 * np.dot(q1.unitized().xyzw, q2.unitized().xyzw) ** 2 - 1, 1))

def distance_box_point(box, point):
    """
    Computes the distance between a box and a point
    """
    if box.contains_point(point):
        return 0.

    return distance_point_point(point, project_point_on_box(box, point))


def project_point_on_box(box, point):
    """
    Projects a point on a box
    """
    return Point(min(max(point[0], box.xmin), box.xmax),
                 min(max(point[1], box.ymin), box.ymax),
                 min(max(point[2], box.zmin), box.zmax))


# # Works when only using horizontal rectangular blocks (with file block.urdf)
# def collision_rectangles(pos, state):
#     if len(state['blocks']) > 0 and ((abs(np.array(state['blocks'])[:,0] - pos[0]) < 0.099) & (abs(np.array(state['blocks'])[:,2] - pos[2]) < 0.049)).any():
#         return True
#     if len(state['obstacles']) > 0 and ((abs(np.array(state['obstacles'])[:,0] - pos[0]) < 0.074) & (abs(np.array(state['obstacles'])[:,2] - pos[2]) < 0.049)).any():
#         return True
#     return False

# def block_vertices_2d(block):
#     for face, vertices in block.face.items():
#         n = block.face_normal(face)
#         if np.abs(n[1] - 1) < 1e-3:
#             break
    
#     for i in vertices:
#         vertex = block.vertex_coordinates(i)
#         # if vertex[1] > 0:
#         yield [vertex[0], vertex[2]]
#     # for vertex in block.vertices():
#     #     point = block.vertex_point(vertex)
#     #     if point[1] > 0:
#     #         yield (point[0], point[2])


def maximum_tension(assembly):
    max_tension = 0.
    for edge in assembly.graph.edges():
        interfaces = assembly.graph.edge_attribute(edge, "interfaces")
        for interface in interfaces:
            n = len(interface.points)
            for i in range(n):
                force = interface.forces[i]
                tension = force['c_np'] - force['c_nn']
                if tension < 0:
                    max_tension = max(max_tension, -tension)
    return max_tension