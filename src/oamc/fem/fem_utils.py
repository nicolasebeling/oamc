import logging
from itertools import product
from typing import Literal

import numpy
from numpy.typing import NDArray

from oamc.enums import ElementType

logger = logging.getLogger(__name__)

N_INT_POINTS: dict[ElementType, int] = {
    ElementType.HEX8: 8,
    ElementType.HEX20: 27,
    # 4-point rule for TET10:
    ElementType.TET10: 4,
    # 11-point rule for TET10:
    # ElementType.TET10: 11,
    ElementType.TET4: 1,
}

A = (5 + 3 * numpy.sqrt(5)) / 20
B = (5 - numpy.sqrt(5)) / 20

INT_POINTS: dict[ElementType, list[tuple[float, float, float]]] = {
    ElementType.HEX8: list(product((-numpy.sqrt(1 / 3), numpy.sqrt(1 / 3)), repeat=3)),
    ElementType.HEX20: list(product((-numpy.sqrt(3 / 5), 0, numpy.sqrt(3 / 5)), repeat=3)),
    # 4-point rule for TET10:
    ElementType.TET10: [
        (B, B, B),
        (A, B, B),
        (B, A, B),
        (B, B, A),
    ],
    # 11-point rule for TET10:
    # ElementType.TET10: [
    #     (0.25, 0.25, 0.25),
    #     (0.78571429, 0.07142857, 0.07142857),
    #     (0.07142857, 0.78571429, 0.07142857),
    #     (0.07142857, 0.07142857, 0.78571429),
    #     (0.07142857, 0.07142857, 0.07142857),
    #     (0.3994036, 0.3994036, 0.1005964),
    #     (0.3994036, 0.1005964, 0.3994036),
    #     (0.1005964, 0.3994036, 0.3994036),
    #     (0.3994036, 0.1005964, 0.1005964),
    #     (0.1005964, 0.3994036, 0.1005964),
    #     (0.1005964, 0.1005964, 0.3994036),
    # ],
    ElementType.TET4: [
        (0.25, 0.25, 0.25),
    ],
}


# Lesson learned: The sum of the integration weights must be the volume of the reference element
# (2 for hexahedral and 1/6 for tetrahedral elements, for example), not 1.


INT_WEIGHTS: dict[ElementType, list[float]] = {
    ElementType.HEX8: [wx * wy * wz for wx, wy, wz in product((1, 1), repeat=3)],
    ElementType.HEX20: [wx * wy * wz for wx, wy, wz in product((5 / 9, 8 / 9, 5 / 9), repeat=3)],
    # 4-point rule for TET10:
    ElementType.TET10: [1 / 24] * 4,
    # 11-point rule for TET10:
    # ElementType.TET10: [-0.01315556] + [0.00762222] * 4 + [0.02488889] * 6,
    ElementType.TET4: [1 / 6],
}


def dof_indices(node_indices: NDArray, dof_per_node: int = 3) -> NDArray:
    """Determine global DOF indices.

    Parameters
    ----------
    node_indices : numpy.ndarray
        Array of node indices.
    dof_per_node : int, default: 3
        Number of degrees of freedom per node.

    Returns
    -------
    numpy.ndarray
        Array of global DOF indices.
    """
    return numpy.repeat(
        node_indices,
        repeats=dof_per_node,
    ) * dof_per_node + numpy.tile(
        numpy.arange(dof_per_node),
        reps=len(node_indices),
    )


def N(
    element_type: ElementType,
    rst: NDArray | tuple[float, float, float],
    cache: bool = False,
) -> NDArray:
    """Compute shape function (N) values.

    Parameters
    ----------
    element_type : ElementType
        Element type.
    rst : numpy.ndarray or tuple of float
        Point in isoparametric natural coordinates (xi, eta, zeta).
    cache : bool, default: False
        Whether to cache the result. Only effective is `rst` is a tuple.

    Returns
    -------
    numpy.ndarray
        Shape function (N) values at the given point.
    """
    if isinstance(rst, tuple) and (element_type, rst) in N.cache:
        return N.cache[(element_type, rst)]

    match element_type:
        case ElementType.HEX8:
            r = rst[0]
            s = rst[1]
            t = rst[2]
            value = (
                numpy.array(
                    [
                        (1 - r) * (1 - s) * (1 - t),
                        (1 + r) * (1 - s) * (1 - t),
                        (1 + r) * (1 + s) * (1 - t),
                        (1 - r) * (1 + s) * (1 - t),
                        (1 - r) * (1 - s) * (1 + t),
                        (1 + r) * (1 - s) * (1 + t),
                        (1 + r) * (1 + s) * (1 + t),
                        (1 - r) * (1 + s) * (1 + t),
                    ]
                )
                / 8
            )
        case ElementType.HEX20:
            r = rst[0]
            s = rst[1]
            t = rst[2]
            value = (
                numpy.array(
                    [
                        # Corner nodes 1–8:
                        (1 - r) * (1 - s) * (1 - t) * ((-r - s - t) - 2) / 2,
                        (1 + r) * (1 - s) * (1 - t) * ((+r - s - t) - 2) / 2,
                        (1 + r) * (1 + s) * (1 - t) * ((+r + s - t) - 2) / 2,
                        (1 - r) * (1 + s) * (1 - t) * ((-r + s - t) - 2) / 2,
                        (1 - r) * (1 - s) * (1 + t) * ((-r - s + t) - 2) / 2,
                        (1 + r) * (1 - s) * (1 + t) * ((+r - s + t) - 2) / 2,
                        (1 + r) * (1 + s) * (1 + t) * ((+r + s + t) - 2) / 2,
                        (1 - r) * (1 + s) * (1 + t) * ((-r + s + t) - 2) / 2,
                        # Midside nodes 9–20:
                        (1 - r**2) * (1 - s) * (1 - t),  # on edge 1-2
                        (1 + r) * (1 - s**2) * (1 - t),  # on edge 2-3
                        (1 - r**2) * (1 + s) * (1 - t),  # on edge 3-4
                        (1 - r) * (1 - s**2) * (1 - t),  # on edge 4-5
                        (1 - r**2) * (1 - s) * (1 + t),  # on edge 5-6
                        (1 + r) * (1 - s**2) * (1 + t),  # on edge 6-7
                        (1 - r**2) * (1 + s) * (1 + t),  # on edge 7-8
                        (1 - r) * (1 - s**2) * (1 + t),  # on edge 8-5
                        (1 - r) * (1 - s) * (1 - t**2),  # on edge 1-5
                        (1 + r) * (1 - s) * (1 - t**2),  # on edge 2-6
                        (1 + r) * (1 + s) * (1 - t**2),  # on edge 3-7
                        (1 - r) * (1 + s) * (1 - t**2),  # on edge 4-8
                    ]
                )
                / 4
            )
        case ElementType.TET10:
            L1, L2, L3, L4 = iso_to_bary(rst)
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
            value = iso_to_bary(rst)
        case _:
            raise ValueError(f"Unknown element type: {element_type}")

    if isinstance(rst, tuple) and cache:
        N.cache[(element_type, rst)] = value

    return value


N.cache = {}


def dN_drst(
    element_type: ElementType,
    rst: NDArray | tuple[float, float, float],
    cache: bool = False,
) -> NDArray:
    """
    Compute shape function (N) gradients with respect to isoparametric
    natural coordinates.

    Parameters
    ----------
    element_type : ElementType
        Element type.
    rst : numpy.ndarray or tuple of float
        Point in isoparametric natural coordinates (xi, eta, zeta).
    cache : bool, default: False
        Whether to cache the result. Only effective if `rst` is a tuple.

    Returns
    -------
    numpy.ndarray
        Shape function (N) gradients with respect to isoparametric
        natural coordinates at the given point.
    """
    if isinstance(rst, tuple) and (element_type, rst) in dN_drst.cache:
        return dN_drst.cache[(element_type, rst)]

    match element_type:
        case ElementType.HEX8:
            r = rst[0]
            s = rst[1]
            t = rst[2]
            value = (
                numpy.array(
                    [
                        [
                            -1 * (1 - s) * (1 - t),
                            (1 - r) * -1 * (1 - t),
                            (1 - r) * (1 - s) * -1,
                        ],
                        [
                            +1 * (1 - s) * (1 - t),
                            (1 + r) * -1 * (1 - t),
                            (1 + r) * (1 - s) * -1,
                        ],
                        [
                            +1 * (1 + s) * (1 - t),
                            (1 + r) * +1 * (1 - t),
                            (1 + r) * (1 + s) * -1,
                        ],
                        [
                            -1 * (1 + s) * (1 - t),
                            (1 - r) * +1 * (1 - t),
                            (1 - r) * (1 + s) * -1,
                        ],
                        [
                            -1 * (1 - s) * (1 + t),
                            (1 - r) * -1 * (1 + t),
                            (1 - r) * (1 - s) * +1,
                        ],
                        [
                            +1 * (1 - s) * (1 + t),
                            (1 + r) * -1 * (1 + t),
                            (1 + r) * (1 - s) * +1,
                        ],
                        [
                            +1 * (1 + s) * (1 + t),
                            (1 + r) * +1 * (1 + t),
                            (1 + r) * (1 + s) * +1,
                        ],
                        [
                            -1 * (1 + s) * (1 + t),
                            (1 - r) * +1 * (1 + t),
                            (1 - r) * (1 + s) * +1,
                        ],
                    ]
                )
                / 8
            )
        case ElementType.HEX20:
            r = rst[0]
            s = rst[1]
            t = rst[2]
            value = numpy.array(
                [
                    # 1
                    [
                        (s - 1) * (t - 1) * (2 * r + s + t + 1) / 8,
                        (r - 1) * (t - 1) * (r + 2 * s + t + 1) / 8,
                        (r - 1) * (s - 1) * (r + s + 2 * t + 1) / 8,
                    ],
                    # 2
                    [
                        (s - 1) * (t - 1) * (2 * r - s - t - 1) / 8,
                        (r + 1) * (t - 1) * (r - 2 * s - t - 1) / 8,
                        (r + 1) * (s - 1) * (r - s - 2 * t - 1) / 8,
                    ],
                    # 3
                    [
                        (s + 1) * (t - 1) * (-2 * r - s + t + 1) / 8,
                        (r + 1) * (t - 1) * (-r - 2 * s + t + 1) / 8,
                        (r + 1) * (s + 1) * (-r - s + 2 * t + 1) / 8,
                    ],
                    # 4
                    [
                        (s + 1) * (t - 1) * (-2 * r + s - t - 1) / 8,
                        (r - 1) * (t - 1) * (-r + 2 * s - t - 1) / 8,
                        (r - 1) * (s + 1) * (-r + s - 2 * t - 1) / 8,
                    ],
                    # 5
                    [
                        (s - 1) * (t + 1) * (-2 * r - s + t - 1) / 8,
                        (r - 1) * (t + 1) * (-r - 2 * s + t - 1) / 8,
                        (r - 1) * (s - 1) * (-r - s + 2 * t - 1) / 8,
                    ],
                    # 6
                    [
                        (s - 1) * (t + 1) * (-2 * r + s - t + 1) / 8,
                        (r + 1) * (t + 1) * (-r + 2 * s - t + 1) / 8,
                        (r + 1) * (s - 1) * (-r + s - 2 * t + 1) / 8,
                    ],
                    # 7
                    [
                        (s + 1) * (t + 1) * (2 * r + s + t - 1) / 8,
                        (r + 1) * (t + 1) * (r + 2 * s + t - 1) / 8,
                        (r + 1) * (s + 1) * (r + s + 2 * t - 1) / 8,
                    ],
                    # 8
                    [
                        (s + 1) * (t + 1) * (2 * r - s - t + 1) / 8,
                        (r - 1) * (t + 1) * (r - 2 * s - t + 1) / 8,
                        (r - 1) * (s + 1) * (r - s - 2 * t + 1) / 8,
                    ],
                    # 9 (edge 1-2)
                    [
                        -r * (s - 1) * (t - 1) / 2,
                        -(r**2 - 1) * (t - 1) / 4,
                        -(r**2 - 1) * (s - 1) / 4,
                    ],
                    # 10 (edge 2-3)
                    [
                        (s**2 - 1) * (t - 1) / 4,
                        s * (r + 1) * (t - 1) / 2,
                        (r + 1) * (s**2 - 1) / 4,
                    ],
                    # 11 (edge 3-4)
                    [
                        r * (s + 1) * (t - 1) / 2,
                        (r**2 - 1) * (t - 1) / 4,
                        (r**2 - 1) * (s + 1) / 4,
                    ],
                    # 12 (edge 4-1)
                    [
                        -(s**2 - 1) * (t - 1) / 4,
                        -s * (r - 1) * (t - 1) / 2,
                        -(r - 1) * (s**2 - 1) / 4,
                    ],
                    # 13 (edge 5-6)
                    [
                        r * (s - 1) * (t + 1) / 2,
                        (r**2 - 1) * (t + 1) / 4,
                        (r**2 - 1) * (s - 1) / 4,
                    ],
                    # 14 (edge 6-7)
                    [
                        -(s**2 - 1) * (t + 1) / 4,
                        -s * (r + 1) * (t + 1) / 2,
                        -(r + 1) * (s**2 - 1) / 4,
                    ],
                    # 15 (edge 7-8)
                    [
                        -r * (s + 1) * (t + 1) / 2,
                        -(r**2 - 1) * (t + 1) / 4,
                        -(r**2 - 1) * (s + 1) / 4,
                    ],
                    # 16 (edge 8-5)
                    [
                        (s**2 - 1) * (t + 1) / 4,
                        s * (r - 1) * (t + 1) / 2,
                        (r - 1) * (s**2 - 1) / 4,
                    ],
                    # 17 (edge 1-5)
                    [
                        -(s - 1) * (t**2 - 1) / 4,
                        -(r - 1) * (t**2 - 1) / 4,
                        -t * (r - 1) * (s - 1) / 2,
                    ],
                    # 18 (edge 2-6)
                    [
                        (s - 1) * (t**2 - 1) / 4,
                        (r + 1) * (t**2 - 1) / 4,
                        t * (r + 1) * (s - 1) / 2,
                    ],
                    # 19 (edge 3-7)
                    [
                        -(s + 1) * (t**2 - 1) / 4,
                        -(r + 1) * (t**2 - 1) / 4,
                        -t * (r + 1) * (s + 1) / 2,
                    ],
                    # 20 (edge 4-8)
                    [
                        (s + 1) * (t**2 - 1) / 4,
                        (r - 1) * (t**2 - 1) / 4,
                        t * (r - 1) * (s + 1) / 2,
                    ],
                ]
            )
        case ElementType.TET10:
            L1, L2, L3, L4 = iso_to_bary(rst)
            value = numpy.array(
                [
                    # 1
                    [1 - 4 * L1, 1 - 4 * L1, 1 - 4 * L1],
                    # 2
                    [4 * L2 - 1, 0, 0],
                    # 3
                    [0, 4 * L3 - 1, 0],
                    # 4
                    [0, 0, 4 * L4 - 1],
                    # 5 (edge 1-2)
                    [4 * (L1 - L2), -4 * L2, -4 * L2],
                    # 6 (edge 2-3)
                    [4 * L3, 4 * L2, 0],
                    # 7 (edge 3-1)
                    [-4 * L3, 4 * (L1 - L3), -4 * L3],
                    # 8 (edge 4-1)
                    [-4 * L4, -4 * L4, 4 * (L1 - L4)],
                    # 9 (edge 4-2)
                    [4 * L4, 0, 4 * L2],
                    # 10 (edge 4-3)
                    [0, 4 * L4, 4 * L3],
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
    Fill the elemental strain-displacement (B) matrix of a 3D solid
    element.

    Parameters
    ----------
    dN_dxyz : numpy.ndarray
        Shape function (N) gradients with respect to global cartesian
        coordinates.

    Returns
    -------
    numpy.ndarray
        Elemental strain-displacement (B) matrix of a 3D solid element.
    """
    if dN_dxyz.shape[1] != 3:
        raise ValueError(
            "Incorrect shape function gradient format. Correct format: [[dN1/dx, dN1/dy, dN1/dz], ...]"
        )

    B = numpy.zeros(
        shape=(6, 3 * dN_dxyz.shape[0]),
        dtype=float,
    )

    for i, grad_N in enumerate(dN_dxyz):
        B[0, 3 * i + 0] = grad_N[0]
        B[1, 3 * i + 1] = grad_N[1]
        B[2, 3 * i + 2] = grad_N[2]
        B[3, 3 * i + 1] = grad_N[2]
        B[3, 3 * i + 2] = grad_N[1]
        B[4, 3 * i + 0] = grad_N[2]
        B[4, 3 * i + 2] = grad_N[0]
        B[5, 3 * i + 0] = grad_N[1]
        B[5, 3 * i + 1] = grad_N[0]

    return B


def equivalent_tensile_stress(s: NDArray) -> NDArray:
    """
    Compute the equivalent tensile (von Mises) stresses from stress
    vectors.

    Parameters
    ----------
    s : numpy.ndarray
        Array of shape (N, 6,) where each row is a stress vector in
        standard Voigt format.

    Returns
    -------
    numpy.ndarray
        Array of shape (N,) containing equivalent tensile (von Mises)
        stress values.
    """
    if s.shape[1] != 6:
        raise ValueError(
            "Incorrect stress vector format. Correct format: [[X, Y, Z, YZ, ZX, XY], ...]"
        )

    s1 = ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) / 2
    s2 = 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    return numpy.sqrt(s1 + s2)


def iso_to_bary(iso: tuple[float, float, float] | NDArray) -> NDArray:
    """
    Transforms isoparametric natural coordinates to barycentric
    coordinates.

    Parameters
    ----------
    iso : numpy.ndarray or tuple of float
        Point in isoparametric natural coordinates.

    Returns
    -------
    numpy.ndarray
        Point in barycentric coordinates.
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


def bary_to_iso(bary: tuple[float, float, float] | NDArray) -> NDArray:
    """
    Transforms barycentric coordinates to isoparametric natural
    coordinates.

    Parameters
    ----------
    bary : numpy.ndarray or tuple of float
        Point in barycentric coordinates.

    Returns
    -------
    numpy.ndarray
        Point in isoparametric natural coordinates.
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
    Build the 6 x 6 stress transformation matrix from a 3 x 3
    rotation matrix.

    Voigt convention: `[11, 22, 33, 23, 13, 12]`

    Parameters
    ----------
    R : numpy.ndarray
        Rotation matrix.
    convention : {"active", "passive"}
        Whether R is an active or passive rotation.

    Returns
    -------
    numpy.ndarray
        Strain transformation matrix of shape (6, 6,).
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
    Build the 6 x 6 strain transformation matrix from a 3 x 3
    rotation matrix.

    Voigt convention: [11, 22, 33, 23, 13, 12]

    Parameters
    ----------
    R : numpy.ndarray
        Rotation matrix.
    convention : {"active", "passive"}
        Whether R is an active or passive rotation.

    Returns
    -------
    numpy.ndarray
        Strain transformation matrix of shape (6, 6,).
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
