"""Load-Based Generation of Fiber Paths for FDM Printing"""

import csv
from pathlib import Path

import numpy

from oamc.core import CompositeMaterial, CompositeModel
from oamc.enums import AngleUnit, ProjectionMethod
from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial
from oamc.integrations.ansys.parser import APDLParser
from oamc.logging import enable_logging
from oamc.post import Viewer

DIR = Path(__file__).parent.resolve()


def main() -> None:
    enable_logging()

    parser = APDLParser(DIR / "ds.dat")

    # Mistake in BSc thesis: Applied -250 instead of -100 N, which resulted in
    # a stress peak in one corner and incorrect fiber paths.
    parser.apply_bearing_load((-100, 0, 0), "HOLE")

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
        title="OAMC Example 3: Load-Based Generation of Fiber Paths for FDM Printing",
    )

    data = []

    c_0 = 4 * model.compliance(model.p)
    data.append(
        {
            "iteration": 0,
            "total fiber length from scalar fields": 0,
            "exact total fiber length": 0,
            "average fiber-filament volume fraction": 0,
            "compliance": round(c_0, 3),
            "fiber efficiency": 0,
            "relative compliance with fiber penalization": 0,
        }
    )

    model.init_q()

    for i in range(1):
        print(f"- Initialization iteration {i + 1} -")

        # model.init_p_by_least_squares(v_min=0.0, v_max=0.6)
        model.init_p_by_least_squares(v_min=0.1, v_max=0.4)

        # model.init_p_by_least_squares(v_min=0.1, v_max=1)
        # model.init_p_by_least_squares(v_min=0.2, v_max=1)
        # model.p = (model.mesh.nodes[:, 1] * 0.35 * 0.078 / model.fiber_area) + 0.25 # for 7.8 % unidirectional

        model.compute_fibers(p_splits=2, min_length=40)

        l = model.precise_total_length
        v = l * model.fiber_area / (4 * model.mesh.volume)
        c = 4 * model.compliance(model.p)
        data.append(
            {
                "iteration": i + 1,
                "total fiber length from scalar fields": round(
                    4 * model.total_length(model.p)[0], 3
                ),
                "exact total fiber length": round(l, 3),
                "average fiber-filament volume fraction": round(v, 3),
                "compliance": round(c, 3),
                "fiber efficiency": round((1 - c / c_0) / v, 3),
                "relative compliance with fiber penalization": round((c / c_0) ** 2 + v**2, 3),
            }
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

    # model.compute_fibers(p_splits=2, min_length=50)

    print(f"Total number of points before downsampling: {model.total_number_of_points}")
    model.downsample_fibers_by_rdp(max_deviation=0.1)
    print(f"Total number of points after downsampling: {model.total_number_of_points}")
    model.remove_outliers(max_length=1, min_angle=numpy.pi / 4)
    print(f"Total number of points after removing outliers: {model.total_number_of_points}")

    model.save_fibers(
        directory=DIR / "example_3_fibers",
        subdirectories=True,
        convention="unit vector",
        angle_unit=AngleUnit.DEG360,
    )

    with open(DIR / "example_3_data.csv", "w", newline="") as csvfile:
        fields = [
            "iteration",
            "total fiber length from scalar fields",
            "exact total fiber length",
            "average fiber-filament volume fraction",
            "compliance",
            "fiber efficiency",
            "relative compliance with fiber penalization",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

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
