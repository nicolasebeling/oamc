"""
Load-Based Generation of Fiber Paths for Nonplanar FDM Printing
---------------------------------------------------------------
...
"""

from oamc.constants import BANNER
from oamc.enums import ProjectionMethod
from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial
from oamc.integrations.ansys.parser import APDLParser
from oamc.logging import enable_logging
from oamc.post import Viewer
from oamc.x import CompositeMaterial, CompositeModel


def main() -> None:
    enable_logging()

    print(BANNER)

    parser = APDLParser(R"C:\Users\nicol\Desktop\TEST01_files\dp0\SYS\MECH\ds.dat")

    # parser = APDLParser(R"C:\Users\nicol\Desktop\full_plate_files\dp0\SYS\MECH\ds.dat")
    # parser.apply_bearing_load(force=(0, -1000, 0), target="HOLE")

    model, model_node_number_to_index, model_elem_number_to_index = parser.get_solid_model()
    mold, mold_node_number_to_index, mold_elem_number_to_index = parser.get_surface_mesh("MOLD")

    matrix_material = IsotropicMaterial(
        E=1300,
        nu=0.39,
        rho=1.14e-9,
    )

    fiber_material = TransverselyIsotropicMaterial(
        E1=66550,
        E2=8400,
        nu12=0.29,
        G23=2340,
        G12=2340,
        rho=1.75e-9,
    )

    model = CompositeModel(
        mesh=model.mesh,
        mold=mold,
        material=CompositeMaterial(
            matrix_material=matrix_material,
            fiber_material=fiber_material,
        ),
        dbc=model.dbc,
        nbc=model.nbc,
        fiber_diameter=0.70,
        layer_height=0.35,
    )

    print(f"Structural compliance before initialization: {round(model.compliance(model.p), 3)} mJ")

    model.init_q()

    model.init_p_by_least_squares(v_min=0.0, v_max=0.5)
    # model.init_p_with_euler_lagrange()

    print(f"Structural compliance after initialization: {round(model.compliance(model.p), 3)} mJ")

    # def callback(*args) -> None:
    #     print(args)

    # model.optimize_p(
    #     max_fiber_length=3e4,
    #     callback=callback,
    #     iteration_limit=10,
    # )

    # print(f"Structural compliance after optimization: {round(model.compliance(model.p), 3)} mJ")

    # model.filter_p_by_diffusion(iterations=10, diffusion_level=0.02)

    # print(f"Structural compliance after filtering: {round(model.compliance(model.p), 3)} mJ")

    l = model._total_length(model.p)[0]
    print(f"Total fiber length: {round(l, 3)} mm")
    print(f"Average fiber volume fraction: {round(l * model.fiber_area / model.mesh.volume, 3)}")

    model.save_fibers(directory="./examples/2/fibers/")

    viewer = Viewer(model)
    viewer.view(
        show_edges=False,
        show_origin=True,
        f_scaling_factor=0.3,
        u_scaling_factor=20.0,
        projection_method=ProjectionMethod.L2,
        opacity=0.3,
        paths=model.fibers,
        title="OAMC Example 2: Load-Based Generation of Fiber Paths for Nonplanar FDM Printing",
    )


if __name__ == "__main__":
    main()
