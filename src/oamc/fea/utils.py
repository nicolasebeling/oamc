import logging
from itertools import product
from typing import Literal

import numpy
from numpy.typing import NDArray

from oamc.enums import ElementType

logger = logging.getLogger(__name__)


INTEGRATION_POINTS = {
    ElementType.HEX8: list(product((-numpy.sqrt(1 / 3), numpy.sqrt(1 / 3)), repeat=3)),
    ElementType.HEX20: list(product((-numpy.sqrt(3 / 5), 0, numpy.sqrt(3 / 5)), repeat=3)),
    ElementType.TET10: [
        (0.13819660, 0.13819660, 0.13819660),
        (0.58541020, 0.13819660, 0.13819660),
        (0.13819660, 0.58541020, 0.13819660),
        (0.13819660, 0.13819660, 0.58541020),
    ],
    ElementType.TET4: [
        (0.25, 0.25, 0.25),
    ],
}


# Lesson learned: The sum of the integration weights must be the volume of the reference element
# (2 for hexahedral and 1/6 for tetrahedral elements, for example), not 1.


INTEGRATION_WEIGHTS = {
    ElementType.HEX8: [wx * wy * wz for wx, wy, wz in product((1, 1), repeat=3)],
    ElementType.HEX20: [wx * wy * wz for wx, wy, wz in product((5 / 9, 8 / 9, 5 / 9), repeat=3)],
    ElementType.TET10: [1 / 24, 1 / 24, 1 / 24, 1 / 24],
    ElementType.TET4: [1 / 6],
}


def N(
    element_type: ElementType,
    x: NDArray | tuple[float, float, float],
    cache: bool = False,
) -> NDArray:
    """
    :param element_type: element type
    :param x: point in isoparametric natural coordinates (xi, eta, zeta)
    :param cache: whether to cache the result
    :return: shape function (N) values at the given point
    """

    if isinstance(x, tuple) and (element_type, x) in N.cache:
        return N.cache[(element_type, x)]

    match element_type:
        case ElementType.HEX8:
            value = (
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
        case ElementType.HEX20:
            value = (
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
        case ElementType.TET10:
            L1, L2, L3, L4 = iso_to_bary(x)
            value = numpy.array(
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
                    4 * L1 * L4,  # on edge 4-1
                    4 * L2 * L4,  # on edge 4-2
                    4 * L3 * L4,  # on edge 4-3
                ]
            )
        case ElementType.TET4:
            value = iso_to_bary(x)
        case _:
            raise ValueError(f"Unknown element type: {element_type}")

    if isinstance(x, tuple) and cache:
        N.cache[(element_type, x)] = value

    return value


N.cache = {}


def dN_drst(
    element_type: ElementType,
    rst: NDArray | tuple[float, float, float],
    cache: bool = False,
) -> NDArray:
    """
    :param element_type: element type
    :param rst: point in isoparametric natural coordinates (xi, eta, zeta)
    :param cache: whether to cache the result (only if rst is a tuple)
    :return: shape function (N) gradients w.r.t. isoparametric natural coordinates
    """

    if isinstance(rst, tuple) and (element_type, rst) in dN_drst.cache:
        return dN_drst.cache[(element_type, rst)]

    match element_type:
        case ElementType.HEX8:
            value = (
                numpy.array(
                    [
                        [
                            -1 * (1 - rst[1]) * (1 - rst[2]),
                            (1 - rst[0]) * -1 * (1 - rst[2]),
                            (1 - rst[0]) * (1 - rst[1]) * -1,
                        ],
                        [
                            +1 * (1 - rst[1]) * (1 - rst[2]),
                            (1 + rst[0]) * -1 * (1 - rst[2]),
                            (1 + rst[0]) * (1 - rst[1]) * -1,
                        ],
                        [
                            +1 * (1 + rst[1]) * (1 - rst[2]),
                            (1 + rst[0]) * +1 * (1 - rst[2]),
                            (1 + rst[0]) * (1 + rst[1]) * -1,
                        ],
                        [
                            -1 * (1 + rst[1]) * (1 - rst[2]),
                            (1 - rst[0]) * +1 * (1 - rst[2]),
                            (1 - rst[0]) * (1 + rst[1]) * -1,
                        ],
                        [
                            -1 * (1 - rst[1]) * (1 + rst[2]),
                            (1 - rst[0]) * -1 * (1 + rst[2]),
                            (1 - rst[0]) * (1 - rst[1]) * +1,
                        ],
                        [
                            +1 * (1 - rst[1]) * (1 + rst[2]),
                            (1 + rst[0]) * -1 * (1 + rst[2]),
                            (1 + rst[0]) * (1 - rst[1]) * +1,
                        ],
                        [
                            +1 * (1 + rst[1]) * (1 + rst[2]),
                            (1 + rst[0]) * +1 * (1 + rst[2]),
                            (1 + rst[0]) * (1 + rst[1]) * +1,
                        ],
                        [
                            -1 * (1 + rst[1]) * (1 + rst[2]),
                            (1 - rst[0]) * +1 * (1 + rst[2]),
                            (1 - rst[0]) * (1 + rst[1]) * +1,
                        ],
                    ]
                )
                / 8
            )
        case ElementType.HEX20:
            raise NotImplementedError(
                "Shape function gradients for HEX20 are not yet implemented."
            )
        case ElementType.TET10:
            L1, L2, L3, L4 = iso_to_bary(rst)
            value = numpy.array(
                [
                    # Corner nodes 1-4:
                    [1 - 4 * L1, 1 - 4 * L1, 1 - 4 * L1],
                    [4 * L2 - 1, 0, 0],
                    [0, 4 * L3 - 1, 0],
                    [0, 0, 4 * L4 - 1],
                    # Midside nodes 5-10:
                    [4 * (L1 - L2), -4 * L2, -4 * L2],  # edge 1-2
                    [4 * L3, 4 * L2, 0],  # edge 2-3
                    [-4 * L3, 4 * (L1 - L3), -4 * L3],  # edge 3-1
                    [-4 * L4, -4 * L4, 4 * (L1 - L4)],  # edge 4-1
                    [4 * L4, 0, 4 * L2],  # edge 4-2
                    [0, 4 * L4, 4 * L3],  # edge 4-3
                ]
            )
        case ElementType.TET4:
            L1, L2, L3, L4 = iso_to_bary(rst)
            value = numpy.array(
                [
                    [-1, -1, -1],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )
        case _:
            raise ValueError(f"Unknown element type: {element_type}")

    if isinstance(rst, tuple) and cache:
        dN_drst.cache[(element_type, rst)] = value

    return value


dN_drst.cache = {}


def B(dN_dxyz: NDArray) -> NDArray:
    """
    :param dN_dxyz: shape function (N) gradients with respect to global cartesian coordinates
    :return: elemental strain-displacement (B) matrix of a 3D solid element
    """
    if dN_dxyz.shape[1] != 3:
        raise ValueError(
            "Incorrect shape function gradient format. Correct format: [[dN1/dx, dN1/dy, dN1/dz], ...]"
        )

    B = numpy.zeros(
        shape=(6, 3 * dN_dxyz.shape[0]),
        dtype=numpy.float64,
    )

    for i, grad_N in enumerate(dN_dxyz):
        B[0, 3 * i + 0] = grad_N[0]
        B[1, 3 * i + 1] = grad_N[1]
        B[2, 3 * i + 2] = grad_N[2]
        B[3, 3 * i + 0] = grad_N[1]
        B[3, 3 * i + 1] = grad_N[0]
        B[4, 3 * i + 1] = grad_N[2]
        B[4, 3 * i + 2] = grad_N[1]
        B[5, 3 * i + 0] = grad_N[2]
        B[5, 3 * i + 2] = grad_N[0]

    return B


def equivalent_tensile_stress(s: NDArray) -> NDArray:
    """Calculates equivalent tensile (von Mises) stresses from stress vectors.

    :param s: N x 6 stress vectors in Ansys format [[X, Y, Z, XY, YZ, ZX], ...]
    :return: N equivalent tensile (von Mises) stresses
    """
    if s.shape[1] != 6:
        raise ValueError(
            "Incorrect stress vector format. Correct format: [[X, Y, Z, XY, YZ, ZX], ...]"
        )

    s1 = ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) / 2
    s2 = 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    return numpy.sqrt(s1 + s2)


def iso_to_bary(iso: tuple | NDArray) -> NDArray:
    """Transforms isoparametric natural coordinates to barycentric coordinates.

    :param iso: point in isoparametric natural coordinates
    :return: point in barycentric coordinates
    """
    if isinstance(iso, tuple):
        length = len(iso)
    elif isinstance(iso, numpy.ndarray):
        length = iso.shape[0]
    else:
        raise ValueError("Isoparametric coordinates must be of type tuple or NDArray.")

    match length:
        case 2:
            return numpy.array([1 - iso[0] - iso[1], iso[0], iso[1]])
        case 3:
            return numpy.array([1 - iso[0] - iso[1] - iso[2], iso[0], iso[1], iso[2]])
        case _:
            raise ValueError(f"Invalid isoparametric coordinates length: {length}")


def bary_to_iso(bary: tuple | NDArray) -> NDArray:
    """Transforms barycentric coordinates to isoparametric natural coordinates.

    :param bary: point in barycentric coordinates
    :return: point in isoparametric natural coordinates
    """
    if isinstance(bary, tuple):
        length = len(bary)
    elif isinstance(bary, numpy.ndarray):
        length = bary.shape[0]
    else:
        raise ValueError("Barycentric coordinates must be of type tuple or NDArray.")

    match length:
        case 3:
            return numpy.array([bary[1], bary[2]])
        case 4:
            return numpy.array([bary[1], bary[2], bary[3]])
        case _:
            raise ValueError(f"Invalid barycentric coordinates length: {length}")


def T_s(
    R: NDArray,
    convention: Literal["active", "passive"] = "passive",
) -> NDArray:
    """
    Build the 6 x 6 stress transformation matrix from a 3x3 rotation.

    Engineering Voigt convention: [11, 22, 33, 23, 13, 12]

    :param R: rotation matrix
    :param convention: whether R is an active or passive rotation
    :return T_s:
    """

    if not numpy.allclose(R @ R.T, numpy.eye(3)):
        raise ValueError("R must be a rotation matrix (orthonormal).")

    if convention == "active":
        R = R.T

    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    T_s = numpy.array(
        [
            [
                r11 * r11,
                r12 * r12,
                r13 * r13,
                2 * r12 * r13,
                2 * r13 * r11,
                2 * r11 * r12,
            ],
            [
                r21 * r21,
                r22 * r22,
                r23 * r23,
                2 * r22 * r23,
                2 * r23 * r21,
                2 * r21 * r22,
            ],
            [
                r31 * r31,
                r32 * r32,
                r33 * r33,
                2 * r32 * r33,
                2 * r33 * r31,
                2 * r31 * r32,
            ],
            [
                r21 * r31,
                r22 * r32,
                r23 * r33,
                r22 * r33 + r23 * r32,
                r23 * r31 + r21 * r33,
                r21 * r32 + r22 * r31,
            ],
            [
                r11 * r31,
                r12 * r32,
                r13 * r33,
                r12 * r33 + r13 * r32,
                r13 * r31 + r11 * r33,
                r11 * r32 + r12 * r31,
            ],
            [
                r11 * r21,
                r12 * r22,
                r13 * r23,
                r12 * r23 + r13 * r22,
                r13 * r21 + r11 * r23,
                r11 * r22 + r12 * r21,
            ],
        ],
        dtype=float,
    )

    return T_s


def T_e(
    R: NDArray,
    convention: Literal["active", "passive"] = "passive",
) -> NDArray:
    """
    Build the 6 x 6 strain transformation matrix from a 3x3 rotation.

    Engineering Voigt convention: [11, 22, 33, 23, 13, 12]

    :param R: rotation matrix
    :param convention: whether R is an active or passive rotation
    :return T_e:
    """

    if not numpy.allclose(R @ R.T, numpy.eye(3)):
        raise ValueError("R must be a rotation matrix (orthonormal).")

    if convention == "active":
        R = R.T

    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    T_e = numpy.array(
        [
            [
                r11 * r11,
                r12 * r12,
                r13 * r13,
                r12 * r13,
                r13 * r11,
                r11 * r12,
            ],
            [
                r21 * r21,
                r22 * r22,
                r23 * r23,
                r22 * r23,
                r23 * r21,
                r21 * r22,
            ],
            [
                r31 * r31,
                r32 * r32,
                r33 * r33,
                r32 * r33,
                r33 * r31,
                r31 * r32,
            ],
            [
                2 * r21 * r31,
                2 * r22 * r32,
                2 * r23 * r33,
                r22 * r33 + r23 * r32,
                r23 * r31 + r21 * r33,
                r21 * r32 + r22 * r31,
            ],
            [
                2 * r11 * r31,
                2 * r12 * r32,
                2 * r13 * r33,
                r12 * r33 + r13 * r32,
                r13 * r31 + r11 * r33,
                r11 * r32 + r12 * r31,
            ],
            [
                2 * r11 * r21,
                2 * r12 * r22,
                2 * r13 * r23,
                r12 * r23 + r13 * r22,
                r13 * r21 + r11 * r23,
                r11 * r22 + r12 * r21,
            ],
        ],
        dtype=float,
    )

    return T_e
