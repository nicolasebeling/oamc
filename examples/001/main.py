from os.path import join

from oamc.fem import Model
from oamc.io import read_model, save_paths
from oamc.lpp import generate_load_paths, plot_load_paths
from oamc.mechanics import Axis

# Example 001:
# Generating load paths from a static structural analysis in Ansys Mechanical

# You can use this example in two ways:
# 1. Run it as is (that is, using the text files provided in this directory).
#    If you are interested in the original Ansys Workbench files, feel free to
#    reach out.
# 2. Open your own static structural analysis in Ansys Mechanical, insert the
#    provided APDL script apdl/export_nodes_elements_stresses.txt (right-click
#    on Solution > Insert > Commands), right-click on it and select Execute
#    Post Commands. This will export
#    - nodes.txt,
#    - elements.txt,
#    - types.txt,
#    - and stresses.txt
#    to the corresponding MECH directory.
#    In addition, you have to provide seed points, either directly as NDArrays
#    or as a named selection of nodes exported as a text file from Mechanical.
#    Then, navigate to the corresponding MECH directory, right-click on the
#    adress bar, select Copy Adress as Text and insert the absolute path here:

MODEL_DIRECTORY = R"./examples/001"  # or absolute path to your MECH directory

if __name__ == "__main__":
    # Create a Model instance from
    # - nodes.txt,
    # - elements.txt,
    # - types.txt,
    # - stresses.txt:
    model: Model = read_model(MODEL_DIRECTORY)

    paths, magnitudes = generate_load_paths(
        part=model,
        # Select the direction for which the load paths shall be generated:
        axis=Axis.MAX,
        # Specify a suitable step size in the length unit of your model:
        step_size=1,
        # Specify seed points as a list of NDArrays or an abolute path to a
        # text file containing seed points exported from Ansys Mechanical:
        seeds=join(MODEL_DIRECTORY, "seeds.txt"),
    )

    save_paths(
        paths=paths,
        # Specify the directory where the paths shall be saved:
        directory=join(MODEL_DIRECTORY, "paths"),
        # Choose the format in which the paths shall be saved:
        format="SpaceClaim",
    )

    plot_load_paths(
        model=model,
        paths=paths,
        magnitudes=magnitudes,
        show_edges=True,
        opacity=0.3,
    )
