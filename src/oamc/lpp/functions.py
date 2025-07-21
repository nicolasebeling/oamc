from time import perf_counter as timer
from typing import Callable

import numpy
from numpy.linalg import eig, norm
from numpy.typing import NDArray
from pyvista import Plotter, PolyData, UnstructuredGrid

from oamc.fem.functions import cell_type, node_count
from oamc.fem.model import Model
from oamc.lpp.rk_method import RKMethod
from oamc.mechanics.axis import Axis
from oamc.mechanics.functions import von_mises_stress


def pointing_stress_vector(stress: NDArray, axis: Axis) -> NDArray:
    """
    Computes the pointing stress vector for a given stress tensor and axis as
    defined in [1].

    :param stress: stress tensor
    :param axis: axis along which the stress vector shall be computed
    :return: pointing stress vector
    """
    match axis:
        case Axis.X:
            s = stress[:, 0]
            return s
        case Axis.Y:
            s = stress[:, 1]
            return s
        case Axis.Z:
            s = stress[:, 2]
            return s
        case Axis.MIN:
            values, vectors = eig(stress)
            return vectors[:, values.argsort()[0]] * values[values.argsort()[0]]
        case Axis.INTERMEDIATE:
            values, vectors = eig(stress)
            return vectors[:, values.argsort()[1]] * values[values.argsort()[1]]
        case Axis.MAX:
            values, vectors = eig(stress)
            return vectors[:, values.argsort()[2]] * values[values.argsort()[2]]
        case _:
            raise ValueError("Unknown axis.")


def rk_step(
    f: Callable[[NDArray], NDArray],
    x: NDArray,
    step_size: float,
    direction: NDArray | None = None,
    method: RKMethod = RKMethod.RK4,
) -> NDArray:
    """
    Computes one Runge-Kutta (RK) integration step.

    :param f: function to integrate
    :param x: current point
    :param step_size: step size for the RK4 method
    :param direction: The direction is chosen such that the dot product of the
        provided direction and the current step is positive. This is required
        when integrating eigenvectors, which may randomly change direction.
    :param method: RK integration scheme
    :return: RK step
    """

    match method:
        case RKMethod.RK4:
            k1 = f(x)
            if direction is not None and direction @ k1 < 0:
                k1 = -k1
            k2 = f(x + step_size * k1 / 2)
            if direction is not None and direction @ k2 < 0:
                k2 = -k2
            k3 = f(x + step_size * k2 / 2)
            if direction is not None and direction @ k3 < 0:
                k3 = -k3
            k4 = f(x + k3)
            if direction is not None and direction @ k4 < 0:
                k4 = -k4
            return step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        case _:
            return ValueError(f"Unknown RK method: {method}")


def generate_load_paths(
    part: Model,
    axis: Axis,
    step_size: float = 1,
    step_limit: int = 10000,
    seeds: list[NDArray] | NDArray | str | None = None,
) -> tuple[list[NDArray], list[NDArray]]:
    """
    :param model: model containing nodes and elements
    :param direction: direction in which the load paths shall be generated
    :param step_size: step size
    :param step_limit: maximum number of steps per path direction
    :param seeds: seed points as a list of 1 x 3 NDArrays, an N x 3 NDArray,
        or a path to a selection of nodes exported from Ansys Mechanical as
        a text file
    :return: load paths as a list of N x 3 NDArrays, pointing stress vector
        magnitudes as a list of N x 1 arrays where N is the number of points
        defining the path
    """

    if seeds is None:
        raise NotImplementedError("Automatic seed generation not yet implemented.")

    if part.stresses is None:
        raise ValueError("Part must contain stress data to generate load paths.")

    if isinstance(seeds, str):
        seeds = numpy.loadtxt(seeds, dtype=float, skiprows=1, usecols=(1, 2, 3))

    start = timer()

    paths = []
    magnitudes = []

    def f(x):
        V = pointing_stress_vector(part.stress(x), axis)
        return V / norm(V)

    for index, seed in enumerate(seeds):
        forward_path = [seed]
        forward_intensity = [numpy.linalg.norm(pointing_stress_vector(part.stress(seed), axis))]
        step = numpy.ones(3)
        try:
            for _ in range(step_limit):
                step = rk_step(
                    f=f,
                    x=forward_path[-1],
                    step_size=step_size,
                    direction=step,
                )
                forward_path.append(forward_path[-1] + step)
                forward_intensity.append(
                    numpy.linalg.norm(pointing_stress_vector(part.stress(forward_path[-1]), axis))
                )
            print(f"Forward path {index + 1} terminated after exceeding maximum number of steps.")
        except ValueError:
            print(f"Forward path {index + 1} terminated after leaving the structure.")
        if len(forward_path) > len(forward_intensity):
            forward_intensity.append(forward_intensity[-1])

        backward_path = [seed]
        backward_intensity = [numpy.linalg.norm(pointing_stress_vector(part.stress(seed), axis))]
        step = -numpy.ones(3)
        try:
            for _ in range(step_limit):
                step = rk_step(
                    f=f,
                    x=backward_path[-1],
                    step_size=step_size,
                    direction=step,
                )
                backward_path.append(backward_path[-1] + step)
                backward_intensity.append(
                    numpy.linalg.norm(pointing_stress_vector(part.stress(backward_path[-1]), axis))
                )
            print(f"Backward path {index + 1} terminated after exceeding maximum number of steps.")
        except ValueError:
            print(f"Backward path {index + 1} terminated after leaving the structure.")
        if len(backward_path) > len(backward_intensity):
            backward_intensity.append(backward_intensity[-1])

        backward_path.reverse()
        backward_intensity.reverse()
        paths.append(numpy.array(backward_path[:-1] + forward_path))
        magnitudes.append(numpy.array(backward_intensity[:-1] + forward_intensity))

    print(f"Paths generated in {round(timer() - start, 3)} seconds.")

    return paths, magnitudes


def plot_load_paths(
    model: Model,
    paths: list[NDArray],
    magnitudes: list[NDArray] | None = None,
    show_edges: bool = True,
    opacity: float = 0.3,
) -> None:
    """
    :param model: model
    :param paths: load paths as a list of N x 3 NDArrays, where N is the number
        of points defining the path
    :param magnitudes: pointing stress vector magnitudes as a list of N x 1
        NDArrays, where N is the number of points defining the path
    :param show_edges: whether to show the edges of the finite elements
    :param opacity: opacity of the model
    """
    start = timer()

    # Each cell (finite element in this context) must start with the number of points (8) followed by the indices of its nodes:
    cells = [[node_count(type)] + list(nodes) for nodes, type in zip(model.elements, model.types)]
    types = [cell_type(type) for type in model.types]
    points = model.nodes
    grid = UnstructuredGrid(cells, types, points)

    # Add von Mises stress as grid point data:
    grid.point_data["Von Mises stress values for body mesh\n"] = von_mises_stress(model.stresses)

    plotter = Plotter(title="Load Paths")

    # Plot part:
    plotter.add_mesh(
        grid,
        scalars="Von Mises stress values for body mesh\n",
        cmap="coolwarm",
        show_edges=show_edges,
        color="lightblue",
        opacity=opacity,
    )

    # Plot load paths:
    for index, (path, magnitude) in enumerate(
        zip(paths, magnitudes if magnitudes is not None else [None] * len(paths))
    ):
        curve = PolyData()
        curve.points = path
        if magnitude is not None:
            curve[f"Pointing Stress Vector Magnitude {index}\n"] = magnitude
        curve.lines = numpy.hstack([[path.shape[0]], numpy.arange(path.shape[0])])
        plotter.add_mesh(
            curve,
            color="black",
            scalars=f"Pointing Stress Vector Magnitude {index}\n"
            if magnitude is not None
            else None,
            show_scalar_bar=False,
            cmap="coolwarm",
            line_width=3,
        )

    # Use parallel projection (no perspective view):
    plotter.parallel_projection = True

    # Add coordinate system in the lower left corner:
    plotter.add_axes()

    print(f"Model and load paths plotted in {round(timer() - start, 3)} seconds.")

    # Show plot:
    plotter.show()
