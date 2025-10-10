import logging
from os import makedirs
from os.path import join
from time import perf_counter as timer
from typing import Callable, Literal

import numpy
from numpy.linalg import eig, norm
from numpy.typing import NDArray
from pyvista import Plotter, PolyData, UnstructuredGrid

from oamc.constants import CELL_TYPE_FROM_ELEMENT_TYPE, NODE_COUNT_FROM_ELEMENT_TYPE
from oamc.enums import Axis, RKMethod
from oamc.fem.analysis import Analysis
from oamc.fem.utils import equivalent_tensile_stress
from oamc.lpp.path import Path

logger = logging.getLogger(__name__)


def pointing_stress_vector(stress: NDArray, axis: Axis) -> NDArray:
    """Compute the pointing stress vector as defined in [1].

    NOTE: The pointing stress vectors are not yet normalized.

    :param stress: stress vector
    :param axis: axis along which the stress vector shall be computed
    :return: pointing stress vector
    """
    tensor = numpy.array(
        [
            [stress[0], stress[5], stress[4]],
            [stress[5], stress[1], stress[3]],
            [stress[4], stress[3], stress[2]],
        ]
    )
    match axis:
        case Axis.X:
            return tensor[:, 0]
        case Axis.Y:
            return tensor[:, 1]
        case Axis.Z:
            return tensor[:, 2]
        case Axis.MIN:
            values, vectors = eig(tensor)
            sorted_indices = values.argsort()
            return vectors[:, sorted_indices[0]] * values[sorted_indices[0]]
        case Axis.INT:
            values, vectors = eig(tensor)
            sorted_indices = values.argsort()
            return vectors[:, sorted_indices[1]] * values[sorted_indices[1]]
        case Axis.MAX:
            values, vectors = eig(tensor)
            sorted_indices = values.argsort()
            return vectors[:, sorted_indices[2]] * values[sorted_indices[2]]
        case _:
            raise ValueError(f"Unknown axis: {axis}")


def RK_step(
    f: Callable[[NDArray], NDArray],
    x: NDArray,
    step_size: float,
    direction: NDArray | None = None,
    method: RKMethod = RKMethod.RK4,
) -> NDArray:
    """Compute one Runge-Kutta (RK) integration step.

    :param f: function to integrate
    :param x: current point
    :param step_size: step size for the RK4 method
    :param direction: The direction is chosen such that the dot product of the
        provided direction and the current step is positive. This is useful
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
            raise ValueError(f"Unknown RK method: {method}")


class LPP:
    """
    Load path plotter (LPP) based on [1].

    Integrate, plot, and export load paths from FEA stress results.

    :param analysis: analysis providing the mesh and stress values
    """

    def __init__(self, analysis: Analysis):
        self.analysis = analysis
        self.paths: list[Path] | None = None

    def generate_load_paths(
        self,
        axis: Axis,
        step_size: float = 1.0,
        step_limit: int = 10000,
        seeds: NDArray | str | None = None,
        seed_selection: list[int] | None = None,
    ) -> list[Path]:
        """
        Generate load paths, store them as an instance variable, and return them.

        :param mesh: mesh containing nodes and elements
        :param step_size: step size
        :param step_limit: maximum number of steps per path direction
        :param seeds: seed points as an N x 3 array or a path to a selection of
            nodes exported from Ansys Mechanical in text format (optional)
        :param seed_selection: selection of seed point indices if `seeds` is a
            file (optional, meant for testing)
        :return: generated load paths as a list of Path objects
        """

        if seeds is None:
            raise NotImplementedError("Automatic seed generation not yet implemented.")

        if isinstance(seeds, str):
            seeds = numpy.loadtxt(seeds, dtype=float, skiprows=1, usecols=(1, 2, 3))
            if seed_selection is not None:
                seeds = seeds[seed_selection]

        start = timer()

        paths = []

        def f(x):
            psv = pointing_stress_vector(self.analysis.stress_at(x), axis)
            return psv / norm(psv)

        for index, seed in enumerate(seeds):
            forward_path = [seed]
            forward_intensity = [norm(pointing_stress_vector(self.analysis.stress_at(seed), axis))]
            step = numpy.ones(3)
            try:
                for _ in range(step_limit):
                    step = RK_step(
                        f=f,
                        x=forward_path[-1],
                        step_size=step_size,
                        direction=step,
                    )
                    forward_path.append(forward_path[-1] + step)
                    forward_intensity.append(
                        norm(
                            pointing_stress_vector(self.analysis.stress_at(forward_path[-1]), axis)
                        )
                    )
                logger.info(
                    f"Forward path {index + 1} terminated after exceeding maximum number of steps."
                )
            except ValueError:
                logger.info(f"Forward path {index + 1} terminated after leaving the structure.")
            if len(forward_path) > len(forward_intensity):
                forward_intensity.append(forward_intensity[-1])

            backward_path = [seed]
            backward_intensity = [
                norm(pointing_stress_vector(self.analysis.stress_at(seed), axis))
            ]
            step = -numpy.ones(3)
            try:
                for _ in range(step_limit):
                    step = RK_step(
                        f=f,
                        x=backward_path[-1],
                        step_size=step_size,
                        direction=step,
                    )
                    backward_path.append(backward_path[-1] + step)
                    backward_intensity.append(
                        norm(
                            pointing_stress_vector(
                                self.analysis.stress_at(backward_path[-1]), axis
                            )
                        )
                    )
                logger.info(
                    f"Backward path {index + 1} terminated after exceeding maximum number of steps."
                )
            except ValueError:
                logger.info(f"Backward path {index + 1} terminated after leaving the structure.")
            if len(backward_path) > len(backward_intensity):
                backward_intensity.append(backward_intensity[-1])

            backward_path.reverse()
            backward_intensity.reverse()
            paths.append(
                Path(
                    coordinates=numpy.array(backward_path[:-1] + forward_path),
                    scalars=numpy.array(backward_intensity[:-1] + forward_intensity),
                )
            )

        logger.info(f"Paths generated in {round(timer() - start, 3)} seconds.")

        self.paths = paths

        return paths

    def plot_paths(
        self,
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

        if self.paths is None:
            raise ValueError("No paths to plot. Generate paths first.")

        start = timer()

        # Each cell (finite element in this context) must start with the number of points (8) followed by the indices of its nodes:
        cells = [
            [NODE_COUNT_FROM_ELEMENT_TYPE[element.etype]] + element.nodes
            for element in self.analysis.mesh.elements
        ]
        types = [
            CELL_TYPE_FROM_ELEMENT_TYPE[element.etype] for element in self.analysis.mesh.elements
        ]
        points = self.analysis.mesh.nodes
        grid = UnstructuredGrid(cells, types, points)

        # Add von Mises stress as grid point data:
        grid.point_data["Von Mises Stress\n"] = equivalent_tensile_stress(self.analysis.s)

        plotter = Plotter(title="Load Path Plotter")

        # Plot part:
        plotter.add_mesh(
            grid,
            scalars="Von Mises Stress\n",
            cmap="coolwarm",
            show_edges=show_edges,
            color="lightblue",
            opacity=opacity,
        )

        # Plot load paths:
        for index, path in enumerate(self.paths):
            curve = PolyData()
            curve.points = path.coordinates
            if path.scalars is not None:
                curve[f"PSV Magnitude {index}\n"] = path.scalars
            curve.lines = numpy.hstack(
                [[path.coordinates.shape[0]], numpy.arange(path.coordinates.shape[0])]
            )
            plotter.add_mesh(
                curve,
                color="black",
                scalars=f"PSV Magnitude {index}\n" if path.scalars is not None else None,
                show_scalar_bar=False,
                cmap="coolwarm",
                line_width=3,
            )

        # Use parallel projection (no perspective view):
        plotter.parallel_projection = True

        # Add coordinate system in the lower left corner:
        plotter.add_axes()

        logger.info(f"Model and load paths plotted in {round(timer() - start, 3)} seconds.")

        # Show plot:
        plotter.show()

    def save_paths(
        self,
        directory: str,
        file_format: Literal["SpaceClaim", "CSV"],
    ) -> None:
        """Save paths to text files.

        https://help.spaceclaim.com/2017.0.0/en/Content/Importing_and_exporting.htm

        :param paths: list of NDarrays representing the load paths
        :param directory: directory where the text files will be saved
        :param file_format: format of the exported paths
        """

        if self.paths is None:
            raise ValueError("No paths to save. Generate paths first.")

        start = timer()

        makedirs(directory, exist_ok=True)

        match file_format:
            case "SpaceClaim":
                for index, path in enumerate(self.paths):
                    with open(join(directory, f"{index + 1}.txt"), "w") as file:
                        numpy.savetxt(
                            fname=file,
                            X=path,
                            fmt="%.3f",
                            delimiter="\t",
                            header="3d=true\npolyline=true\n",
                            comments="",
                        )
            case "CSV":
                for index, path in enumerate(self.paths):
                    with open(join(directory, f"{index + 1}.csv"), "w") as file:
                        numpy.savetxt(
                            fname=file,
                            X=path,
                            fmt="%.3f",
                            delimiter=",",
                        )
            case _:
                raise ValueError(f"Unknown path format: {file_format}")

        logger.info(f"Paths saved to {directory} in {round(timer() - start, 3)} seconds.")
