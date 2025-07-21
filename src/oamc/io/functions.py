from os import makedirs
from os.path import join
from time import perf_counter as timer
from typing import Literal

import numpy
from numpy.typing import NDArray

from oamc.fem.ansys_element_type import AnsysElementType
from oamc.fem.model import Model


def read_model(directory: str) -> Model:
    """
    Read a model from text files exported from Ansys Mechanical.

    The following files must be present in the directory:
    - nodes.txt (nodal coordinates)
    - elements.txt (element connectivity)
    - types.txt (element types)
    - stresses.txt (nodal stress values)

    NOTE: All node and element indices are decremented by 1 to convert from one-based to zero-based indexing.

    :param directory: directory containing the aforementioned files
    :return: model
    """

    start = timer()

    # Read nodes:

    nodes = []
    read = False
    with open(join(directory, "nodes.txt")) as file:
        for line in file:
            if read and line.strip():
                nodes.append(line.split()[1:4])
            elif line.strip().startswith("NODE"):
                read = True
    nodes = numpy.array(nodes, dtype=float)

    # Read element types:

    type_dict = {}
    with open(join(directory, "types.txt")) as file:
        for line in file:
            if line.strip().startswith("ELEM"):
                line = line.split()
                match line[4]:
                    case "SOLID185":
                        type_dict[int(line[2])] = AnsysElementType.SOLID185
                    case "SOLID186":
                        # NOTE: Quadratic elements are currently treated as
                        # linear elements because Ansys does not provide
                        # accurate stress results at midiside nodes. The
                        # exported midiside stresses are simply the mean of
                        # the adjacent corner stresses (that is, linearly
                        # interpolated).
                        type_dict[int(line[2])] = AnsysElementType.SOLID185
                    case "SOLID187":
                        # NOTE: Quadratic elements are currently treated as
                        # linear elements because Ansys does not provide
                        # accurate stress results at midiside nodes. The
                        # exported midiside stresses are simply the mean of
                        # the adjacent corner stresses (that is, linearly
                        # interpolated).
                        type_dict[int(line[2])] = AnsysElementType.SOLID285
                    case "SOLID285":
                        type_dict[int(line[2])] = AnsysElementType.SOLID285
                    case "SURF154":
                        pass  # used for load application
                    case "CONTA174":
                        pass  # used for multi-body parts
                    case "TARGE170":
                        pass  # used for multi-body parts
                    case _:
                        try:
                            spaces = int(line[4])
                            for i in range(list(type_dict)[-1], spaces):
                                type_dict[i + 1] = type_dict[list(type_dict)[-1]]
                        except ValueError:
                            raise ValueError(f"Unknown element type: {line[4]}")

    # Read elements:

    types = []
    elements = []
    read = False
    with open(join(directory, "elements.txt")) as file:
        for line in file:
            if read and line.strip():
                spaces = 0
                for char in line:
                    if char == " ":
                        spaces += 1
                    else:
                        break
                if spaces < 28:
                    types.append(type_dict[int(line.split()[2])])
                    match types[-1]:
                        case AnsysElementType.SOLID185 | AnsysElementType.SOLID186:
                            elements.append([int(n) - 1 for n in line.split()[6:12]])
                        case AnsysElementType.SOLID187:
                            elements.append([int(n) - 1 for n in line.split()[6:14]])
                        case AnsysElementType.SOLID285:
                            elements.append([int(n) - 1 for n in line.split()[6:10]])
                        case _:
                            raise ValueError(f"Unknown element type: {types[-1]}")
                else:
                    match types[-1]:
                        case AnsysElementType.SOLID186:
                            elements[-1].extend([int(n) - 1 for n in line.split()])
                        case AnsysElementType.SOLID187:
                            elements[-1].extend([int(n) - 1 for n in line.split()])
                        case _:
                            pass
            elif line.strip().startswith("ELEM"):
                read = True
    elements = numpy.array(elements, dtype=int)

    # Read stresses:

    stresses = []
    read = False
    with open(join(directory, "stresses.txt")) as file:
        for line in file:
            if read and line.strip():
                stresses.append(line.split()[1:7])
            elif line.strip().startswith("NODE"):
                read = True
    stresses = numpy.array(stresses, dtype=float)

    print(f"Files read in {round(timer() - start, 3)} seconds.")

    return Model(nodes, elements, types, stresses)


def save_paths(paths: list[NDArray], directory: str, format: Literal["SpaceClaim", "CSV"]):
    """
    Save paths to text files formatted for import as 3D polylines in Ansys SpaceClaim.

    :param paths: list of NDarrays representing the load paths
    :param directory: directory where the text files will be saved
    """
    start = timer()
    makedirs(directory, exist_ok=True)
    match format:
        case "SpaceClaim":
            for index, path in enumerate(paths):
                with open(join(directory, f"{index + 1:03}.txt"), "w") as file:
                    numpy.savetxt(
                        fname=file,
                        X=path,
                        fmt="%.3f",
                        delimiter="\t",
                        header="3d=true\npolyline=true\n",
                        comments="",
                    )
        case "CSV":
            for index, path in enumerate(paths):
                with open(join(directory, f"{index + 1:03}.csv"), "w") as file:
                    numpy.savetxt(
                        fname=file,
                        X=path,
                        fmt="%.3f",
                        delimiter=",",
                    )
        case _:
            raise ValueError(f"Unknown path format: {format}")
    print(f"Paths saved to {directory} in {round(timer() - start, 3)} seconds.")
