"""
Classes
-------
LPP
    Load path plotter.
"""

import logging
from os import makedirs
from os.path import join
from time import perf_counter as timer
from typing import Callable, Literal

import numpy
from numpy.typing import NDArray

from oamc.enums import Direction, RKMethod
from oamc.fem.model import SolidModel
from oamc.fiber import Fiber
from oamc.utils import mechanics

logger = logging.getLogger(__name__)


def pointing_stress_vector(stress_vector: NDArray, direction: Direction) -> NDArray:
    """Compute the pointing stress vector as defined in [1].

    Parameters
    ----------
    stress : numpy.ndarray
        Stress vector in Voigt notation.
    direction : oamc.enums.Direction
        Direction along which the stress vector shall be computed.

    Returns
    -------
    numpy.ndarray
        Pointing stress vector.

    Notes
    -----
    The pointing stress vectors are not yet normalized.
    """
    stress_tensor = mechanics.vector_to_tensor(stress_vector)
    match direction:
        case Direction.X:
            return stress_tensor[:, 0]
        case Direction.Y:
            return stress_tensor[:, 1]
        case Direction.Z:
            return stress_tensor[:, 2]
        case Direction.MIN | Direction.INT | Direction.MAX:
            value, vector = mechanics.principal_stress(
                stress_tensor=stress_tensor,
                direction=direction,
            )
            return vector * value
        case _:
            raise ValueError(f"Unknown axis: {direction}")


def integration_step(
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
    Load path plotter (LPP) based on [1]_.

    Integrate and export load paths from FEA stress results. See notes
    for visualization.

    Attributes
    ----------
    model : oamc.fem.model.Model
        FE model providing the mesh and stress values.
    paths : list of oamc.path.Path
        Generated load paths.

    Notes
    -----
    In order to make the `oamc` package as modular as possible, the
    visualization itself has been moved to the `Viewer` class in the
    `oamc.post` subpackage. Generate load paths with this class and
    visualize them with the `Viewer` class.

    References
    ----------
    .. [1] D. Kelly, C. Reidsema, A. Bassandeh, G. Pearce, and M. Lee, "On interpreting load paths and identifying a load bearing topology from finite element analysis," Finite Elements in Analysis and Design, vol. 47, no. 8, pp. 867-876, Aug. 2011, doi: https://doi.org/10.1016/j.finel.2011.03.007.
    """

    def __init__(self, model: SolidModel):
        """
        Attributes
        ----------
        model : oamc.fem.model.Model
            FE model providing the mesh and stress values.
        """
        self.model = model
        self.paths: list[Fiber] = []

    def generate_load_paths(
        self,
        direction: Direction,
        step_size: float = 1.0,
        step_limit: int = 10000,
        seeds: NDArray | str | None = None,
        seed_selection: list[int] | None = None,
    ) -> list[Fiber]:
        """
        Generate load paths, store them as an instance variable, and
        return them.

        Parameters
        ----------
        axis : oamc.enums.Direction
            Direction along which the load paths shall be generated.
        step_size : float, default: 1.0
            Step size for the path integration in the length unit of
            the model.
        step_limit : int, default: 10000
            Maximum number of steps per path and direction.
        seeds : numpy.ndarray or str, optional
            Seed points as an N x 3 array or a path to a selection of
            nodes exported from Ansys Mechanical in text format.
        seed_selection : list of int
            Selection of seed point indices. This is meant for testing.

        Returns
        -------
        list of oamc.path.Path
            Generated load paths.
        """

        if seeds is None:
            raise NotImplementedError("Automatic seed generation not yet implemented.")

        if isinstance(seeds, str):
            seeds = numpy.loadtxt(seeds, dtype=float, skiprows=1, usecols=(1, 2, 3))

        if seed_selection is not None:
            seeds = seeds[seed_selection]

        start = timer()

        def f(x):
            psv = pointing_stress_vector(
                stress_vector=self.model.get_stress_at_point(x),
                direction=direction,
            )
            return psv / numpy.linalg.norm(psv)

        for index, seed in enumerate(seeds):
            forward_path = [seed]
            forward_intensity = [
                numpy.linalg.norm(
                    pointing_stress_vector(
                        self.model.get_stress_at_point(seed),
                        direction,
                    ),
                )
            ]
            step = numpy.ones(3)
            try:
                for _ in range(step_limit):
                    step = integration_step(
                        f=f,
                        x=forward_path[-1],
                        step_size=step_size,
                        direction=step,
                    )
                    forward_path.append(forward_path[-1] + step)
                    forward_intensity.append(
                        numpy.linalg.norm(
                            pointing_stress_vector(
                                self.model.get_stress_at_point(forward_path[-1]),
                                direction,
                            )
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
                numpy.linalg.norm(
                    pointing_stress_vector(
                        self.model.get_stress_at_point(seed),
                        direction,
                    ),
                )
            ]
            step = -numpy.ones(3)
            try:
                for _ in range(step_limit):
                    step = integration_step(
                        f=f,
                        x=backward_path[-1],
                        step_size=step_size,
                        direction=step,
                    )
                    backward_path.append(backward_path[-1] + step)
                    backward_intensity.append(
                        numpy.linalg.norm(
                            pointing_stress_vector(
                                self.model.get_stress_at_point(backward_path[-1]),
                                direction,
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
            self.paths.append(
                Fiber(
                    points=numpy.array(backward_path[:-1] + forward_path),
                    scalar_name="Pointing Stress Vector Magnitude",
                    scalar_values=numpy.array(backward_intensity[:-1] + forward_intensity),
                )
            )

        logger.info(f"Load paths generated in {round(timer() - start, 3)} seconds.")

        return self.paths

    def save_load_paths(
        self,
        directory: str,
        file_format: Literal["SpaceClaim", "CSV"],
    ) -> None:
        """Save load paths to text files.

        https://help.spaceclaim.com/2017.0.0/en/Content/Importing_and_exporting.htm

        Parameters
        ----------
        directory : str
            Path to the directory where the text files will be saved.
        file_format : {"SpaceClaim", "CSV"}
            Format of the exported paths.
        """

        if self.paths is None:
            raise ValueError("No paths to save. Generate load paths first.")

        start = timer()

        makedirs(directory, exist_ok=True)

        match file_format:
            case "SpaceClaim":
                for index, path in enumerate(self.paths):
                    with open(join(directory, f"{index + 1}.txt"), "w") as file:
                        numpy.savetxt(
                            fname=file,
                            X=path.points,
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
                            X=path.points,
                            fmt="%.3f",
                            delimiter=",",
                        )
            case _:
                raise ValueError(f"Unknown path format: {file_format}")

        logger.info(f"Paths saved to {directory} in {round(timer() - start, 3)} seconds.")
