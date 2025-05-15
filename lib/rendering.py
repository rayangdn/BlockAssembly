import matplotlib.pyplot as plt
import numpy as np
import torch



def plot_block(block, ax, facecolor='tab:blue', label=None, face_numbers=False, face_numbers_offset=0.05, face_numbers_color='k', face_numbers_fontsize=12):
    vertices = np.array([[block.vertex_coordinates(v)[0], block.vertex_coordinates(v)[2]] for v in block.vertices_2d()])
    ax.fill(vertices[:, 0], vertices[:, 1], '+', edgecolor='k', facecolor=facecolor)
    if label is not None:
        ax.text(block.centroid()[0], block.centroid()[2], label, ha="center", va="center", color="w")

    if face_numbers:
        for f in block.faces_2d():
            face_center = block.face_centroid(f)
            face_normal = block.face_normal(f)

            # plot face number slightly offset from the center
            ax.text(face_center[0] + face_numbers_offset * face_normal[0], face_center[2] + face_numbers_offset * face_normal[2], f, fontsize=face_numbers_fontsize, color=face_numbers_color, ha='center', va='center')


def plot_task(task, fig, ax):
    if task.obstacles is not None:
        for obstacle in task.obstacles:
            plot_block(obstacle, ax, facecolor='tab:red')
        
    # for target in task.targets:
        # ax.scatter(target.location[0], target.location[-1], marker="*", s=100, color="tab:red" if target.fixed else "tab:green")
    for location in task.targets:
        ax.scatter(location[0], location[-1], marker="*", s=100, color="tab:green")


def plot_assembly_env(assembly, fig=None, ax=None, plot_forces=False, force_scale=1.0, plot_edges=False, equal=False, face_numbers=False, nodes=False, task=None):
    """
    Plot the CRA assembly in 2D with forces.
    """

    # if assembly.__class__.__name__ == "AssemblyEnv":
    #     bounds = assembly.bounds
    #     assembly = assembly.cra_assembly

    # elif assembly.__class__.__name__ == "AssemblyGym":
    #     bounds = assembly.assembly_env.bounds
    #     assembly = assembly.assembly_env.cra_assembly

    # if graph is None:
    graph = assembly.graph
    if fig is None:
        fig, ax = plt.subplots(figsize=(5,5) if equal else None)

    # plot blocks
    for j, (i, node) in enumerate(graph.node.items()):
        block = node['block']
        facecolor = 'tab:blue'
        if i == -1:
            facecolor = 'gray'
        elif node.get('is_support'):
            facecolor = 'tab:orange'
        
        plot_block(block, ax, facecolor=facecolor, label=str(j), face_numbers=face_numbers, face_numbers_offset=-0.15, face_numbers_fontsize=6, face_numbers_color='white')

        if i == -1:
            for f, attrs in block.faces_2d(data=True):
                face_frame = block.face_frame_2d(f)

                # for offset in attrs["offsets_x"]:
                #     p = face_frame.to_world_coordinates(Point(offset, 0, 0))
                #     ax.scatter(p[0], p[2], color='k', marker='x')

    # plot nodes
    if nodes:
        for node in graph.nodes():
            point = assembly.node_point(node)
            ax.plot(point[0], point[2], 'o', color='tab:red')

    # plot edges
    if plot_edges:
        for edge in graph.edges():
            u, v = edge
            point_u = assembly.node_point(u)
            point_v = assembly.node_point(v)
            ax.plot([point_u[0], point_v[0]], [point_u[2], point_v[2]], 'k--', linewidth=1)

    # plot interfaces
    for interface in assembly.interfaces():
        points = [p for p in interface.points if p[1] > 0]
        points = np.array(points)

        if len(points) > 0:
            ax.plot(points[:, 0], points[:, 2], 'k-' ,linewidth=4)

    # plot obstacles
    if task is not None:
        for i, b in enumerate(assembly.task.obstacles):
            plot_block(b, ax, facecolor='tab:red', label=str(i))

    # plot forces
    if plot_forces:
        for edge in graph.edges():
            interfaces = graph.edge_attribute(edge, "interfaces")
            for interface in interfaces:
                frame = interface.frame

                n = len(interface.points)
                for i in range(n):
                    # plot point
                    point = interface.points[i]
                    if point[1] < 0:
                        continue

                    ax.plot(point[0], point[2], 'o', color='tab:green')

                    force = interface.forces[i]

                    force_vector = [force['c_u'], force['c_v'], force['c_np'] - force['c_nn']]
                    # to world coordinates
                    force_vector = frame.to_world_coordinates(force_vector) - frame.point
                    ax.arrow(point[0], point[2], -force_scale * force_vector[0], -force_scale * force_vector[2], color='tab:green')
    
    if equal:
        ax.axis('equal')

    if task is not None:
        plot_task(task, fig, ax)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='tab:blue', edgecolor='k', label='Placed Blocks'),
        Patch(facecolor='tab:red', edgecolor='k', label='Obstacles'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='tab:green', 
               markersize=15, label='Target Positions')
    ]
    
    # Add support blocks to legend if they exist
    # if any(node.get('is_support', False) for _, node in graph.node.items()):
    #     legend_elements.append(Patch(facecolor='tab:orange', edgecolor='k', label='Support Blocks'))
    
    # Add grey ground block to legend if it exists
    if -1 in graph.node:
        legend_elements.append(Patch(facecolor='gray', edgecolor='k', label='Ground'))
    
    # Add forces to legend if they're being plotted
    if plot_forces:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor='tab:green', markersize=10, label='Force Points'))
    
    ax.legend(handles=legend_elements, loc='best', fontsize=20)

    # bounds = assembly.bounds
    # if bounds is not None:
    #     ax.set_xlim(assembly)
    #     ax.set_ylim(bounds[0][2], bounds[1][2])
    return fig, ax



def render_block_2d(block, xlim, zlim, img_size=(512,512), device='cpu', dtype=torch.float):
    X, Y = np.meshgrid(np.linspace(*xlim, img_size[0]), np.linspace(zlim[1], zlim[0], img_size[1]))
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    return torch.tensor(block.contains_2d_convex(positions).reshape(img_size), device=device).to(dtype)