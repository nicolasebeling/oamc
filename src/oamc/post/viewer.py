"""Contains the ``Viewer`` class."""

import json
import logging
from copy import deepcopy
from pathlib import Path
from time import perf_counter as clock

import numpy
import pyvista

from oamc.enums import Direction, ProjectionMethod
from oamc.fem.model import SolidModel
from oamc.fiber import Fiber
from oamc.utils.mechanics import equivalent_tensile_stress, principal_stress, vector_to_tensor

logger = logging.getLogger(__name__)

VON_MISES_STRESS = "Von Mises Stress"
MAJOR_PRINCIPAL_STRESS = "Major Principal Stress"
MAJOR_PRINCIPAL_STRESS_TITLE = f"{MAJOR_PRINCIPAL_STRESS}\n"


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
        self.title = title
        self.plotter = pyvista.Plotter(title=title)

    @staticmethod
    def view_cached(directory: str | Path) -> bool:
        """Show a previously saved visualization if it exists.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory containing a scene saved by :meth:`view`.

        Returns
        -------
        bool
            ``True`` if a cached scene was shown, otherwise ``False``.
        """

        directory = Path(directory)
        scene_path = directory / "scene.vtm"
        settings_path = directory / "settings.json"

        if not scene_path.is_file() or not settings_path.is_file():
            return False

        with settings_path.open(encoding="utf-8") as file:
            settings = json.load(file)

        print(
            f"Viewing cached scene from {directory.resolve()}. "
            "Delete or replace this cache to update it."
        )

        scene = pyvista.read(scene_path)
        plotter = pyvista.Plotter(title=settings["title"])
        model_scalars = settings["model_scalars"].strip()
        if model_scalars not in scene["model"].array_names:
            model_scalars += " "
        plotter.add_mesh(
            scene["model"],
            scalars=model_scalars,
            cmap="coolwarm",
            show_edges=settings["show_edges"],
            color="lightblue",
            opacity=settings["opacity"],
            scalar_bar_args={"title": MAJOR_PRINCIPAL_STRESS_TITLE},
        )

        if settings["show_forces"]:
            plotter.add_mesh(scene["forces"], color="red")

        for i, scalar_name in enumerate(settings["path_scalar_names"]):
            plotter.add_mesh(
                scene[f"path_{i}"],
                color="grey",
                scalars=scalar_name,
                show_scalar_bar=False,
                cmap="coolwarm",
                line_width=3,
            )

        plotter.parallel_projection = True
        plotter.set_background("white")
        plotter.add_axes()
        if settings["show_origin"]:
            plotter.add_axes_at_origin(labels_off=True)
        plotter.show()

        return True

    def view(
        self,
        show_edges: bool = True,
        show_origin: bool = True,
        f_scaling_factor: float = 0,
        u_scaling_factor: float = 0,
        projection_method: ProjectionMethod = ProjectionMethod.L2,
        opacity: float = 0.5,
        paths: list[Fiber] | None = None,
        cache_directory: str | Path | None = None,
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
        cache_directory : str or pathlib.Path, optional
            Directory in which to save the VTK scene for later replay with
            :meth:`view_cached`.
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
        stress = self.model.get_stress_at_nodes(projection_method=projection_method)
        grid.point_data[VON_MISES_STRESS] = equivalent_tensile_stress(stress)
        grid.point_data[MAJOR_PRINCIPAL_STRESS] = numpy.array(
            [
                principal_stress(
                    stress_tensor=vector_to_tensor(vector=s),
                    direction=Direction.MAX,
                )[0]
                for s in stress
            ]
        )

        scene = pyvista.MultiBlock()
        scene["model"] = grid

        # Plot part:
        self.plotter.add_mesh(
            grid,
            # scalars=VON_MISES_STRESS,
            scalars=MAJOR_PRINCIPAL_STRESS,
            # cmap="spring",
            cmap="coolwarm",
            show_edges=show_edges,
            color="lightblue",
            opacity=opacity,
            scalar_bar_args={"title": MAJOR_PRINCIPAL_STRESS_TITLE},
            # clim=(0, 50),
        )

        # Plot nodal force vector:
        show_forces = f_scaling_factor != 0
        if show_forces:
            f = self.model.f.reshape(-1, 3) * f_scaling_factor
            force_points = pyvista.PolyData(grid.points - f)
            force_points["vectors"] = f
            forces = force_points.glyph(orient="vectors", scale="vectors", factor=1.0)
            scene["forces"] = forces
            self.plotter.add_mesh(forces, color="red")

        path_scalar_names = []
        if paths is not None:
            # Displace paths:
            if u_scaling_factor != 0:
                paths = deepcopy(paths)
                for path in paths:
                    u = []
                    for point in path.points:
                        u.append(self.model.get_u_at_point(point))
                    path.points += numpy.array(u) * u_scaling_factor

            # Plot paths:
            # colors = ["red", "blue", "green", "yellow", "purple"]
            for i, path in enumerate(paths):
                polydata = path.polydata
                scene[f"path_{i}"] = polydata
                path_scalar_names.append(path.scalar_name)
                self.plotter.add_mesh(
                    mesh=polydata,
                    # color=colors[i % 5],
                    color="grey",
                    scalars=path.scalar_name,
                    show_scalar_bar=False,
                    cmap="coolwarm",
                    line_width=3,
                )

        if cache_directory is not None:
            cache_directory = Path(cache_directory)
            cache_directory.mkdir(parents=True, exist_ok=True)
            scene.save(cache_directory / "scene.vtm")
            settings = {
                "title": self.title,
                "model_scalars": MAJOR_PRINCIPAL_STRESS,
                "show_edges": show_edges,
                "show_origin": show_origin,
                "opacity": opacity,
                "show_forces": show_forces,
                "path_scalar_names": path_scalar_names,
            }
            with (cache_directory / "settings.json").open("w", encoding="utf-8") as file:
                json.dump(settings, file, indent=2)

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
