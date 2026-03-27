"""Load-Based Generation of Fiber Paths for FDM Printing"""

from collections import defaultdict
from pathlib import Path

import numpy

from oamc.core import CompositeMaterial, CompositeModel
from oamc.enums import AngleUnit, Direction, ProjectionMethod
from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial
from oamc.fiber import Fiber
from oamc.integrations.ansys.parser import APDLParser
from oamc.logging import enable_logging
from oamc.lpp import LPP
from oamc.post import Viewer

DIR = Path(__file__).parent.resolve()


def main() -> None:
    enable_logging()

    parser = APDLParser(DIR / "ds.dat")

    model = parser.get_solid_model()[0]
    mold = parser.get_surface_mesh("MOLD")[0]

    matrix_material = IsotropicMaterial(
        E=2990,
        nu=0.39,
        rho=1.27e-9,
    )

    fiber_material = TransverselyIsotropicMaterial(
        E1=66550,
        E2=4900,
        nu12=0.32,
        G23=1680,
        G12=1880,
        rho=1.44e-9,
    )

    composite_material = CompositeMaterial(
        matrix_material=matrix_material,
        fiber_material=fiber_material,
    )

    model = CompositeModel(
        mesh=model.mesh,
        mold=mold,
        material=composite_material,
        dbc=model.dbc,
        nbc=model.nbc,
        fiber_diameter=0.70,
        layer_height=0.35,
    )

    viewer = Viewer(
        model=model,
        title="OAMC Example 2: Load-Based Generation of Fiber Paths for FDM Printing",
    )

    # ---------------------------------------------------------------------------------------------

    C_0 = model.compliance(model.p)
    print(f"Structural compliance before initialization: {round(C_0, 3)} mJ")

    model.init_q()

    # for i in range(10):  # for 10.5 %
    for i in range(5):  # for 30.1 %
    # for i in range(1):  # for 9.6 % unidirectional
        print(f"- Initialization iteration {i + 1} -")
        # model.init_p_by_least_squares(v_min=0.0, v_max=1.0)  # for 10.5 %
        model.init_p_by_least_squares(v_min=0.2, v_max=1.0)  # for 30.1 %
        # model.p = (
        #     model.mesh.nodes[:, 1] * 0.35 * 0.1 / model.fiber_area
        # )  # for 9.6 % unidirectional

        model.compute_fibers(p_splits=2, min_length=40)

        L_f = model.precise_total_length
        V_f = L_f * model.fiber_area / model.mesh.volume
        print(f"Structural compliance = {round(model.compliance(model.p), 3)} mJ")
        print(
            f"Total fiber length from scalar fields = {round(model.total_length(model.p)[0], 3)} mm"
        )
        print(f"Precise total fiber length = {round(L_f, 3)} mm")
        print(
            f"Total fiber weight = {round(L_f * model.fiber_area * fiber_material.rho * 1e6, 3)} g"
        )
        print(f"Average fiber volume fraction = {round(V_f, 3)}")
        print(
            f"eta_f = (1 - C / C_0) / V_f = {round((1 - model.compliance(model.p) / C_0) / V_f, 3)}"
        )
        print(
            f"c_p = (C / C_0)**2 + 1 * V_f**2 = {round((model.compliance(model.p) / C_0) ** 2 + V_f**2, 3)}"
        )

    # def callback(*args) -> None:
    #     print(args)

    # model.optimize_p(
    #     max_fiber_length=1e4,
    #     callback=callback,
    #     iteration_limit=5,
    # )

    # print(f"Structural compliance after optimization: {round(model.compliance(model.p), 3)} mJ")

    # model.filter_p_by_diffusion(iterations=10, diffusion_level=0.01)

    # print(f"Structural compliance after diffusion filtering: {round(model.compliance(model.p), 3)} mJ")

    model.compute_fibers(p_splits=2, min_length=40)

    # ---------------------------------------------------------------------------------------------

    # lpp = LPP(model)

    # lpp.generate_load_paths(
    #     # Select the direction for which the load paths shall be generated:
    #     direction=Direction.MAX,
    #     # Specify a suitable step size in the length unit of your model:
    #     step_size=1,
    #     # Specify seed points as an NDArray or an absolute path to a text file
    #     # containing seed points exported from Ansys Mechanical:
    #     seeds=numpy.array(
    #         (
    #             (50, -19, 1.4),
    #             (50, -17, 1.4),
    #             (50, -15, 1.4),
    #             (50, -13, 1.4),
    #             (50, -11, 1.4),
    #             (50, -9, 1.4),
    #             (50, -7, 1.4),
    #             (50, -5, 1.4),
    #             (50, -3, 1.4),
    #             (50, 3, 1.4),
    #             (50, 5, 1.4),
    #             (50, 7, 1.4),
    #             (50, 9, 1.4),
    #             (50, 11, 1.4),
    #             (50, 13, 1.4),
    #             (50, 15, 1.4),
    #             (50, 17, 1.4),
    #             (50, 19, 1.4),
    #         )
    #     ),
    # )

    # paths = lpp.paths
    # fibers = defaultdict(list)

    # for i, path in enumerate(paths):
    #     for level in range(4):
    #         if (level % 2 == i % 2) or True:
    #             points = path.points.copy()
    #             # points[:, 2] = model.layer_height * (0.5 + 2 * level)
    #             points[:, 2] = 1.4

    #             orientations = numpy.zeros_like(points)
    #             orientations[:, 2] = -1

    #             width = model.fiber_area / model.layer_height
    #             dims = (width, model.layer_height)

    #             fibers[level + 1].append(Fiber(points=points, orientations=orientations, dims=dims, scalar_name=path.scalar_name, scalar_values=path.scalar_values))

    # model.fibers = fibers

    # L_f = model.precise_total_length
    # V_f = L_f * model.fiber_area / model.mesh.volume
    # print(f"Precise total fiber length = {round(L_f, 3)} mm")
    # print(f"Total fiber weight = {round(L_f * model.fiber_area * fiber_material.rho * 1e6, 3)} g")
    # print(f"Average fiber volume fraction = {round(V_f, 3)}")

    # ---------------------------------------------------------------------------------------------

    print(f"Total number of points before downsampling: {model.total_number_of_points}")
    model.downsample_fibers_by_rdp(max_deviation=0.05)
    print(f"Total number of points after downsampling: {model.total_number_of_points}")
    model.remove_outliers(max_length=10, min_angle=numpy.pi / 4)
    print(f"Total number of points after removing outliers: {model.total_number_of_points}")

    model.save_fibers(
        directory=DIR / "fibers",
        subdirectories=True,
        convention="unit vector",
        angle_unit=AngleUnit.DEG360,
    )

    viewer.view(
        show_edges=False,
        show_origin=False,
        f_scaling_factor=0,
        u_scaling_factor=0,
        projection_method=ProjectionMethod.L2,
        opacity=0.3,
        paths=model.fibers_as_list,
    )


if __name__ == "__main__":
    main()
