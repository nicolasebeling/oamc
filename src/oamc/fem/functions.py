import numpy
from numpy.typing import NDArray
from pyvista import CellType

from oamc.fem.ansys_element_type import AnsysElementType


def shape_functions(type: AnsysElementType, x: NDArray) -> NDArray:
    """
    :param type: element type
    :param x: point in isoparametric natural coordinates (xi, eta, zeta)
    :return: shape function values at the given point
    """
    match type:
        case AnsysElementType.SOLID185:
            return (
                numpy.array(
                    [
                        (1 - x[0]) * (1 - x[1]) * (1 - x[2]),
                        (1 + x[0]) * (1 - x[1]) * (1 - x[2]),
                        (1 + x[0]) * (1 + x[1]) * (1 - x[2]),
                        (1 - x[0]) * (1 + x[1]) * (1 - x[2]),
                        (1 - x[0]) * (1 - x[1]) * (1 + x[2]),
                        (1 + x[0]) * (1 - x[1]) * (1 + x[2]),
                        (1 + x[0]) * (1 + x[1]) * (1 + x[2]),
                        (1 - x[0]) * (1 + x[1]) * (1 + x[2]),
                    ]
                )
                / 8
            )
        case AnsysElementType.SOLID186:
            return (
                numpy.array(
                    [
                        # Corner nodes 1–8:
                        (1 - x[0]) * (1 - x[1]) * (1 - x[2]) * ((-x[0] - x[1] - x[2]) - 2) / 2,
                        (1 + x[0]) * (1 - x[1]) * (1 - x[2]) * ((+x[0] - x[1] - x[2]) - 2) / 2,
                        (1 + x[0]) * (1 + x[1]) * (1 - x[2]) * ((+x[0] + x[1] - x[2]) - 2) / 2,
                        (1 - x[0]) * (1 + x[1]) * (1 - x[2]) * ((-x[0] + x[1] - x[2]) - 2) / 2,
                        (1 - x[0]) * (1 - x[1]) * (1 + x[2]) * ((-x[0] - x[1] + x[2]) - 2) / 2,
                        (1 + x[0]) * (1 - x[1]) * (1 + x[2]) * ((+x[0] - x[1] + x[2]) - 2) / 2,
                        (1 + x[0]) * (1 + x[1]) * (1 + x[2]) * ((+x[0] + x[1] + x[2]) - 2) / 2,
                        (1 - x[0]) * (1 + x[1]) * (1 + x[2]) * ((-x[0] + x[1] + x[2]) - 2) / 2,
                        # Midside nodes 9–20:
                        (1 - x[0] ** 2) * (1 - x[1]) * (1 - x[2]),  # on edge 1-2
                        (1 + x[0]) * (1 - x[1] ** 2) * (1 - x[2]),  # on edge 2-3
                        (1 - x[0] ** 2) * (1 + x[1]) * (1 - x[2]),  # on edge 3-4
                        (1 - x[0]) * (1 - x[1] ** 2) * (1 - x[2]),  # on edge 4-5
                        (1 - x[0] ** 2) * (1 - x[1]) * (1 + x[2]),  # on edge 5-6
                        (1 + x[0]) * (1 - x[1] ** 2) * (1 + x[2]),  # on edge 6-7
                        (1 - x[0] ** 2) * (1 + x[1]) * (1 + x[2]),  # on edge 7-8
                        (1 - x[0]) * (1 - x[1] ** 2) * (1 + x[2]),  # on edge 8-5
                        (1 - x[0]) * (1 - x[1]) * (1 - x[2] ** 2),  # on edge 1-5
                        (1 + x[0]) * (1 - x[1]) * (1 - x[2] ** 2),  # on edge 2-6
                        (1 + x[0]) * (1 + x[1]) * (1 - x[2] ** 2),  # on edge 3-7
                        (1 - x[0]) * (1 + x[1]) * (1 - x[2] ** 2),  # on edge 4-8
                    ]
                )
                / 4
            )
        case AnsysElementType.SOLID187:
            # Convert isoparametric to barycentric coordinates:
            L1 = 1 - x[0] - x[1] - x[2]
            L2 = x[0]
            L3 = x[1]
            L4 = x[2]
            return numpy.array(
                [
                    # Corner nodes 1-4:
                    L1 * (2 * L1 - 1),
                    L2 * (2 * L2 - 1),
                    L3 * (2 * L3 - 1),
                    L4 * (2 * L4 - 1),
                    # Midside nodes 5-12:
                    4 * L1 * L2,  # on edge 1-2
                    4 * L2 * L3,  # on edge 2-3
                    4 * L3 * L1,  # on edge 3-1
                    4 * L4 * L1,  # on edge 4-1
                    4 * L4 * L2,  # on edge 4-2
                    4 * L4 * L3,  # on edge 4-3
                ]
            )
        case AnsysElementType.SOLID285:
            # Convert isoparametric to barycentric coordinates:
            L1 = 1 - x[0] - x[1] - x[2]
            L2 = x[0]
            L3 = x[1]
            L4 = x[2]
            return numpy.array([L1, L2, L3, L4])
        case _:
            raise ValueError(f"Unknown element type: {type}")


def node_count(type: AnsysElementType) -> int:
    """
    :param type: element type
    :return: number of nodes for the given element type
    """
    match type:
        case AnsysElementType.SOLID185:
            return 8
        case AnsysElementType.SOLID186:
            return 20
        case AnsysElementType.SOLID187:
            return 10
        case AnsysElementType.SOLID285:
            return 4
        case _:
            raise ValueError(f"Unknown element type: {type}")


def cell_type(type: AnsysElementType) -> CellType:
    """
    :param type: element type
    :return: corresponding PyVista cell type
    """
    match type:
        case AnsysElementType.SOLID185:
            return CellType.HEXAHEDRON
        case AnsysElementType.SOLID186:
            return CellType.QUADRATIC_HEXAHEDRON
        case AnsysElementType.SOLID187:
            return CellType.QUADRATIC_TETRA
        case AnsysElementType.SOLID285:
            return CellType.TETRA
        case _:
            raise ValueError(f"Unknown element type: {type}")
