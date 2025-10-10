"""
Generating Load Paths from a Static Structural Analysis in Ansys Mechanical

There are two ways to use this example:

1.  Run it as is (that is, using the files provided in this directory).

2.  Navigate to the MECH directory of your own static structural analysis in
    Ansys Mechanical, select ds.dat, right-click on it, select "Copy as path",
    and assign it to the constant DS_DAT below.

    In addition, you have to provide seed points:

    A.  Export a named selection of nodes from Ansys Mechanical and assign the
        corresponding path to the constant SEEDS_TXT below.
    B.  Alternatively, you can provide the seed points directly as an N x 3
        NDArray to the function LPP.generate_load_paths below.

The generated load paths will be saved to PATH_DIR and visualized with VTK.
"""

from oamc.constants import BANNER
from oamc.enums import Axis
from oamc.fem import Analysis, DSReader
from oamc.logging import enable_logging
from oamc.lpp import LPP

DS_DAT = R"./examples/1/ds.dat"
SEEDS_TXT = R"./examples/1/seeds.txt"
PATH_DIR = R"./examples/1/paths/"


def main() -> None:
    enable_logging()

    print(BANNER)

    with DSReader(DS_DAT) as reader:
        analysis: Analysis = reader.get_analysis()

    lpp = LPP(analysis)

    lpp.generate_load_paths(
        # Select the direction for which the load paths shall be generated:
        axis=Axis.MAX,
        # Specify a suitable step size in the length unit of your model:
        step_size=1,
        # Specify seed points as an NDArray or an absolute path to a text file
        # containing seed points exported from Ansys Mechanical:
        seeds=SEEDS_TXT,
        # To reduce the execution time for testing:
        seed_selection=[0, 100, 200, 300],
    )

    lpp.save_paths(
        # Specify the directory where the paths shall be saved:
        directory=PATH_DIR,
        # Choose the format in which the paths shall be saved:
        file_format="SpaceClaim",
    )

    lpp.plot_paths(
        show_edges=True,
        opacity=0.3,
    )


if __name__ == "__main__":
    main()
