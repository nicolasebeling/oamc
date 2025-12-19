"""Generating Load Paths from a Static Structural Analysis in Ansys

There are two ways to use this example:

1.  Run it as is (that is, using the files provided in this directory).

2.  Navigate to the MECH directory of your own static structural
    analysis in Ansys Mechanical, select ds.dat, right-click on it,
    select "Copy as path", and assign it to the constant DS_DAT below.

    In addition, you have to provide seed points:

    A.  Export a named selection of nodes from Ansys Mechanical and
        assign the corresponding path to the constant SEEDS_TXT below.
    B.  Alternatively, you can provide the seed points directly as an
        N x 3 NDArray to the function LPP.generate_load_paths below.

The generated load paths will be saved to PATH_DIR/paths/ and visualized
with PyVista (a VTK wrapper).
"""

from pathlib import Path

from oamc.enums import Direction, ProjectionMethod
from oamc.integrations.ansys.parser import APDLParser
from oamc.logging import enable_logging
from oamc.lpp import LPP
from oamc.post import Viewer

DIR = Path(__file__).parent.resolve()

DS_DAT = DIR / "ds.dat"
SEEDS_TXT = DIR /  "seeds.txt"
PATH_DIR = DIR / "paths/"


def main() -> None:
    enable_logging()

    # Create a parser object:
    parser = APDLParser(DS_DAT)

    # Manually apply the bearing load to all elements in the named
    # selection "HOLE" because bearing loads cannot be parsed yet:
    parser.apply_bearing_load(force=(0, -1000, 0), target="HOLE")

    # Translate the APDL model to an OAMC model:
    model, node_number_to_index, elem_number_to_index = parser.get_solid_model()

    # Get seed node numbers from named selection in Ansys Mechanical:
    seed_numbers = parser.get_named_selection("SEEDS")
    # Translate seed node numbers to indices:
    seed_indices = [node_number_to_index[number] for number in seed_numbers]
    # Get seed nodes as an NDArray:
    seeds = model.mesh.nodes[seed_indices]

    # Create a load path plotter object:
    lpp = LPP(model)

    lpp.generate_load_paths(
        # Select the direction for which the load paths shall be generated:
        direction=Direction.MAX,
        # Specify a suitable step size in the length unit of your model:
        step_size=1,
        # Specify seed points as an NDArray or an absolute path to a text file
        # containing seed points exported from Ansys Mechanical:
        seeds=seeds,
        # seeds=SEEDS_TXT,
        # To reduce the execution time for testing:
        # seed_selection=[],
    )

    lpp.save_load_paths(
        # Specify the directory where the paths shall be saved:
        directory=PATH_DIR,
        # Choose the format in which the paths shall be saved:
        file_format="SpaceClaim",
    )

    # Create a viewer object:
    viewer = Viewer(
        model=model,
        title="OAMC Example 1: Generating Load Paths from a Static Structural Analysis in Ansys",
    )

    # View the model and load paths:
    viewer.view(
        show_edges=True,
        f_scaling_factor=0.1,
        u_scaling_factor=0.0,
        projection_method=ProjectionMethod.ANSYS,
        opacity=0.3,
        paths=lpp.paths,
    )


if __name__ == "__main__":
    main()
