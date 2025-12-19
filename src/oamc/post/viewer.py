"""
Classes
-------
Viewer
"""

import logging
from copy import deepcopy
from time import perf_counter as clock

import numpy
import pyvista

from oamc.enums import ProjectionMethod
from oamc.fem.model import SolidModel
from oamc.fem.utils import equivalent_tensile_stress
from oamc.fiber import Fiber

logger = logging.getLogger(__name__)


class Viewer:
    def __init__(self, model: SolidModel, title: str = "OAMC Viewer"):
        """
        Parameters
        ----------
        model : oamc.fem.SolidModel
            Finite-element model.
        title : str, default: "OAMC Viewer"
            Title of the PyVista plotter.
        """
        self.model = model
        self.plotter = pyvista.Plotter(title=title)

    def view(
        self,
        show_edges: bool = True,
        show_origin: bool = True,
        f_scaling_factor: float = 0,
        u_scaling_factor: float = 0,
        projection_method: ProjectionMethod = ProjectionMethod.L2,
        opacity: float = 0.5,
        paths: list[Fiber] | None = None,
    ) -> None:
        """Visualize the model in an interactive plot.

        Parameters
        ----------
        show_edges : bool, default: True
            Whether to show the edges of the mesh as lines.
        show_origin : bool, default: True
            Whether to show a coordinate system at the origin.
        f_scaling_factor : float, default: 0
            Scaling factor for the visualization of the equivalent nodal
            force vector. 1 means an arrow length of one length unit per
            force unit and 0 means no visualization, for example.
        u_scaling_factor : float, default: 0
            Scaling factor for the deformation of the part. 0 means no
            deformation.
        projection_method : oamc.enums.ProjectionMethod, default: oamc.enums.ProjectionMethod.L2
            Which method to use to project stress values from
            integration points to nodes.
        opacity : float, default: 0.5
            Opacity of the part.
        paths : list of oamc.path.Fiber
            Paths to plot (currently only instances of oamc.path.Fiber,
            more general in the future).
        """

        start = clock()

        # If the Ansys projection method (linear extrapolation) is used, use a linear copy of the
        # grid, because stresses are only extrapolated to nodes in this case:
        match projection_method:
            case ProjectionMethod.L2:
                grid = self.model.get_grid(
                    u_scaling_factor=u_scaling_factor,
                )
            case ProjectionMethod.ANSYS:
                grid = self.model.get_grid(
                    u_scaling_factor=u_scaling_factor,
                ).linear_copy()
            case _:
                raise ValueError(f"Unkown projection method: {projection_method}")

        # Add von Mises stress as grid point data:
        grid.point_data["Von Mises Stress\n"] = equivalent_tensile_stress(
            self.model.get_stress_at_nodes(projection_method=projection_method)
        )

        # Plot part:
        self.plotter.add_mesh(
            grid,
            scalars="Von Mises Stress\n",
            cmap="coolwarm",
            show_edges=show_edges,
            color="lightblue",
            opacity=opacity,
        )

        # Plot nodal force vector:
        if f_scaling_factor != 0:
            f = self.model.f.reshape(-1, 3) * f_scaling_factor
            self.plotter.add_arrows(
                cent=grid.points - f,
                direction=f,
                color="red",
            )

        # Displace fibers:
        if u_scaling_factor != 0:
            paths = deepcopy(paths)
            for fiber in paths:
                u = []
                for point in fiber.points:
                    u.append(self.model.get_u_at_point(point))
                fiber.points += numpy.array(u) * u_scaling_factor

        # Plot paths:
        colors = ["red", "blue", "green", "yellow", "purple"]
        for i, fiber in enumerate(paths):
            self.plotter.add_mesh(
                mesh=fiber.polydata,
                color=colors[i % 5],
                # color="grey",
                scalars=fiber.scalar_name,
                show_scalar_bar=False,
                cmap="coolwarm",
                line_width=3,
            )

        # Use parallel projection (no perspective view):
        self.plotter.parallel_projection = True

        # Set background:
        self.plotter.set_background("white")

        # Add coordinate system in the lower left corner:
        self.plotter.add_axes()

        # Add coordinate system at the origin:
        if show_origin:
            self.plotter.add_axes_at_origin(labels_off=True)

        logger.info(f"Model plotted in {clock() - start:.2f} seconds.")

        # Show plot:
        self.plotter.show()
