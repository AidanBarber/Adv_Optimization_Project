import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from typing import List, Tuple


def classify_path_points(path: List[Tuple[int, int, int]]) -> \
        Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    if not path:
        return [], []

    # Separate into ground and flying points
    ground_points = []
    flying_points = []

    for point in path:
        z, y, x = point
        if z == 0:  # Ground level
            ground_points.append(point)
        else:  # Flying level
            flying_points.append(point)

    return ground_points, flying_points


def visualize_path_2d(path: List[Tuple[int, int, int]], occ_map, start, goal, title):
    if not path:
        fig, ax = plt.subplots(figsize=(6, 5))
        plt.imshow(occ_map, cmap='gray_r',
                   interpolation='none', origin='lower')
        plt.title("No Path Found")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.imshow(occ_map[0, :, :], cmap='gray_r',
               interpolation='none', origin='lower')

    ground_points, flying_points = classify_path_points(path)

    # Plot ground points
    if ground_points:
        _, ground_ys, ground_xs = zip(*ground_points)
        ax.scatter(ground_xs, ground_ys, color='red', s=30, marker='o',
                   label='Ground Path', alpha=0.8, zorder=3)

    # Plot flying points
    if flying_points:
        _, flying_ys, flying_xs = zip(*flying_points)
        ax.scatter(flying_xs, flying_ys, color='blue', s=30, marker='s',
                   label='Flying Path', alpha=0.8, zorder=3)

    # Plot path
    _, ys, xs = zip(*path)
    ax.plot(xs, ys, color='gray', linewidth=2,
            alpha=1, linestyle='-', zorder=2)

    # Plot start and goal
    _, y, x = start
    ax.plot(x, y, 'go',
            markersize=10, label='Start', zorder=4)
    _, y, x = goal
    ax.plot(x, y, 'r*',
            markersize=14, label='Goal', zorder=4)

    major_ticks = np.arange(0.5, 20, 5)
    minor_ticks = np.arange(0.5, 20, 1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    ax.grid(True, alpha=0.3)
    ax.grid(which='minor', alpha=0.2)
    plt.show()


def visualize_path_3d(path, occ_map):
    plotter = pv.Plotter(window_size=[1600, 1200])

    obs_y, obs_x = np.where(occ_map[0] == 1)
    for x, y in zip(obs_x, obs_y):
        cube = pv.Cube(center=(x, y, 0.25),
                       x_length=0.9, y_length=0.9, z_length=0.5)
        plotter.add_mesh(cube, color="darkgray",
                         opacity=0.6, show_edges=True)

    pts = np.array([el[::-1] for el in path])
    spline = pv.MultipleLines(points=pts)
    plotter.add_mesh(spline, color="royalblue", line_width=6)

    plotter.add_mesh(pv.Sphere(radius=0.2, center=pts[0]), color="green")
    plotter.add_mesh(pv.Sphere(radius=0.3, center=pts[-1]), color="red")

    n_gridlines = 21
    g_values = np.linspace(-0.5, 19.5, n_gridlines)  # Range for x-axis
    for x in g_values:
        line = pv.Line(pointa=(x, -1, 0), pointb=(x, 20, 0))
        plotter.add_mesh(line, color="gray", line_width=1)

    for y in g_values:
        line = pv.Line(pointa=(-1, y, 0), pointb=(20, y, 0))
        plotter.add_mesh(line, color="gray", line_width=1)

    plotter.set_background("white")
    plotter.show_grid(
        n_xlabels=2,
        n_ylabels=2,
        n_zlabels=2,
        grid=True
    )
    plotter.camera_position = 'xz'
    plotter.camera.elevation = 35
    plotter.camera.azimuth = 45
    plotter.show(title="3D Path Visualization")


def visualize_path(path, occ_map, start, goal, title):
    visualize_path_2d(path, occ_map, start, goal, title)
    visualize_path_3d(path, occ_map)
