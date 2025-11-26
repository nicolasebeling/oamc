"""Contains the Fiber class."""

import logging
from dataclasses import dataclass

import numpy
import pyvista
from numpy.typing import NDArray
from typing_extensions import Literal

from oamc.vtk_utils import rectangular_tube

logger = logging.getLogger()


@dataclass(slots=True)
class Fiber:
    """
    Attributes
    ----------
    points : numpy.ndarray
        Points defining the fiber path as an array of shape (number of points, 3).
    orientations : numpy.ndarray, optional
        Orientation of the tool head as unit vectors in an array of the same shape as `points`.
    scalar_name : str, optional
        Name of the scalar values.
    scalar_values : numpy.ndarray, optional
        Scalar values such as stress data.
    dims : tuple of float, optional
        Cross-sectional dimensions of the fiber. Must be either (radius,) or (width, height). In
        the latter case, the `orientation` vectors are used to define the local height direction.
        Hence, `orientation` must be given in this case.
    """

    points: NDArray
    orientations: NDArray | None = None
    scalar_name: str | None = None
    scalar_values: NDArray | None = None
    dims: tuple[float, ...] | None = None

    @property
    def polydata(self):
        """Create a pyvista.PolyData object to visualize the path."""
        if self.dims is None:
            mesh = pyvista.lines_from_points(self.points)
            if self.scalar_name is not None and self.scalar_values is not None:
                mesh[self.scalar_name] = self.scalar_values
            return mesh

        if len(self.dims) == 1:
            mesh = pyvista.lines_from_points(self.points).tube(self.dims[0])
            if self.scalar_name is not None and self.scalar_values is not None:
                mesh[self.scalar_name] = self.scalar_values
            return mesh

        if len(self.dims) == 2:
            if self.orientations is None or self.orientations.shape != self.points.shape:
                raise ValueError(
                    "Rectangular fiber cross sections require an orientation "
                    "array of the same shape as the coordinates array."
                )
            mesh = rectangular_tube(
                p=self.points,
                z=self.orientations,
                w=self.dims[0],
                h=self.dims[1],
            )
            if self.scalar_name is not None and self.scalar_values is not None:
                mesh[self.scalar_name] = self.scalar_values
            return mesh

        raise ValueError("Dimensions must be None or a tuple of length 1 or 2.")

    def save(
        self,
        file: str,
        convention: Literal["XYZ"] = "XYZ",
        angle_unit: Literal["deg", "rad"] = "rad",
        delimiter: str = ",",
        decimals: int = 5,
    ) -> None:
        """Save paths and tool head orientations in CSV format.

        The local z-axis points in the direction of the tool head. The
        local x-axis is the local fiber tangent direction projected onto
        the plane orthogonal to the local z-axis. The local y-axis forms
        a right-handed cartesian coordinate system with the other two
        axes.

        Parameters
        ----------
        file : str
            Path to the file to be saved.
        convention : {"XYZ"}
            Euler angle convention (see [1]_). Capital letters indicate global axes.
        angle_unit : {"rad", "deg"}
            Unit of the exported angles.
        delimiter : str, default: ","
            Delimiter to use in the CSV file.
        decimals : int, default: 3
            Number of decimal places to save.

        References
        ----------
        .. [1] Wikipedia Contributors, "Euler angles," Wikipedia, Oct. 8, 2019. https://en.wikipedia.org/wiki/Euler_angles
        .. [2] A. Owen-Hill, "Robot Euler Angles: The Essential Primer," RoboDK blog, Apr. 03, 2018. https://robodk.com/blog/robot-euler-angles/
        .. [3] "Basic Guide - RoboDK Documentation," Robodk.com, 2015. https://robodk.com/doc/en/Basic-Guide.html#RefFrames
        """

        # Normalize the local z-axes (tool head directions):
        z = self.orientations / numpy.linalg.norm(self.orientations, axis=1, keepdims=True)

        # Determine local x-directions (tangent directions):
        x = numpy.zeros_like(self.points)
        x[0] = self.points[1] - self.points[0]
        x[-1] = self.points[-1] - self.points[-2]
        # Tangent and point (i) = direction of vector from point (i - 1)
        # to point (i + 1) for i in (1, n):
        x[1:-1] = self.points[2:] - self.points[:-2]
        # Normalize the local x-axes:
        x /= numpy.linalg.norm(x, axis=1, keepdims=True)
        # Ensure that the local x-axes are perpendicular to the local z-axes:
        x -= numpy.sum(x * z, axis=1, keepdims=True) * z
        # Again, normalize the local x-axes:
        nx = numpy.linalg.norm(x, axis=1, keepdims=True)
        if numpy.any(nx < 1e-9):
            raise ValueError(
                "The local z-axes must not be aligned with the local tangent directions."
            )
        x /= nx

        # Determine the local y-directions:
        y = numpy.cross(z, x)

        # Assemble rotation matrices as an array of shape (number of points, 3, 3):
        r = numpy.column_stack((x.ravel(), y.ravel(), z.ravel())).reshape(-1, 3, 3)

        # Compute Euler angles from rotation matrices:
        angles = numpy.zeros_like(self.points)
        match convention:
            case "XYZ":
                for i, ri in enumerate(r):
                    angles[i, 1] = numpy.arcsin(-ri[2, 0])  # beta
                    if abs(abs(angles[i, 1]) - numpy.pi) > 1e-9:  # beta is not close to pi:
                        angles[i, 0] = numpy.arctan2(ri[2, 1], ri[2, 2])  # gamma
                        angles[i, 2] = numpy.arctan2(ri[1, 0], ri[0, 0])  # alpha
                    else:  # beta close is to pi, gimbal lock:
                        angles[i, 0] = numpy.arctan2(-ri[1, 2], ri[1, 1])  # gamma
                        # alpha is arbitrary (0 in this case).
                    if round(angles[i, 0], 3) != 0 or round(angles[i, 1], 3) != 0:
                        raise Exception()
            case _:
                raise NotImplementedError(
                    f"Euler angle convention {convention} is not yet implemented."
                )

        match angle_unit:
            case "rad":
                pass
            case "deg":
                numpy.rad2deg(angles, out=angles)
            case _:
                raise NotImplementedError(f"Unknown angle unit: {angle_unit}")

        numpy.savetxt(
            fname=file,
            X=numpy.hstack((self.points, angles)),
            fmt=f"%.{decimals}f",
            delimiter=delimiter,
        )
