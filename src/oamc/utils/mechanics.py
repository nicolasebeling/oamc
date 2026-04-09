"""Utility functions for continuum mechanics.

Functions
---------
vector_to_tensor
tensor_to_vector
tensor_to_matrix
principal_stress
"""

from typing import Literal

import numpy
from numpy.typing import NDArray

from oamc.enums import Direction


def vector_to_tensor(vector: NDArray) -> NDArray:
    return numpy.array(
        [
            [vector[0], vector[5], vector[4]],
            [vector[5], vector[1], vector[3]],
            [vector[4], vector[3], vector[2]],
        ]
    )


def tensor_to_vector(tensor: NDArray) -> NDArray:
    if not numpy.allclose(tensor, tensor.T):
        raise ValueError("Only symmetric tensors can be converted to Voigt notation.")
    return numpy.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            tensor[1, 2],
            tensor[0, 2],
            tensor[0, 1],
        ]
    )


def tensor_to_matrix(
    tensor: NDArray,
    convention: Literal["voigt", "engineering", "mandel"] = "engineering",
) -> NDArray:
    """Convert a fourth-order tensor of shape (3, 3, 3, 3) to an
    equivalent matrix of shape (6, 6).

    Parameters
    ----------
    tensor : numpy.ndarray
        Fourth-order tensor of shape (3, 3, 3, 3).
    convention : {"voigt", "engineering", "mandel"}
        Convention for matrix notation.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (3, 3).
    """

    if tensor.shape != (3, 3, 3, 3):
        raise ValueError("Tensor must have shape (3,3,3,3).")

    index_pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    matrix = numpy.zeros((6, 6), dtype=float)
    for ij, (i, j) in enumerate(index_pairs):
        for kl, (k, l) in enumerate(index_pairs):
            if k == l:
                matrix[ij, kl] = tensor[i, j, k, l]
            else:
                matrix[ij, kl] = tensor[i, j, k, l] + tensor[i, j, l, k]

    if convention.lower() == "voigt":
        return matrix

    if convention.lower() == "engineering":
        factor = numpy.array(
            [1, 1, 1, 2, 2, 2],
            dtype=float,
        )
        return matrix / factor[None, :]

    if convention.lower() == "mandel":
        factor = numpy.array(
            [1, 1, 1, numpy.sqrt(2), numpy.sqrt(2), numpy.sqrt(2)],
            dtype=float,
        )
        return factor[:, None] * matrix / factor[None, :]

    raise ValueError("Unsupported convention.")


def principal_stress(stress_tensor: NDArray, direction: Direction) -> tuple[float, NDArray]:
    """Compute the principal stress in the given direction from the
    given stress tensor.

    Parameters
    ----------
    stress_tensor : numpy.ndarray
        Symmetric stress tensor of shape (3, 3).
    direction : oamc.enums.Direction
        Principal stress direction.

    Returns
    -------
    float
        Principal stress value.
    numpy.ndarray
        Principal stress direction as a unit vector of shape (3,).
    """
    values, vectors = numpy.linalg.eig(stress_tensor)
    sorted_indices = values.argsort()
    match direction:
        case Direction.MIN:
            return values[sorted_indices[0]], vectors[:, sorted_indices[0]]
        case Direction.INT:
            return values[sorted_indices[1]], vectors[:, sorted_indices[1]]
        case Direction.MAX:
            return values[sorted_indices[2]], vectors[:, sorted_indices[2]]
        case _:
            raise ValueError(f"Unknown axis: {direction}")


def equivalent_tensile_stress(s: NDArray) -> NDArray:
    """Compute the equivalent tensile (von Mises) stresses from stress
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


def T_s(
    R: NDArray,
    convention: Literal["active", "passive"] = "passive",
) -> NDArray:
    """Build the 6 x 6 stress transformation matrix from a 3 x 3
    rotation matrix.

    Voigt convention: ``[11, 22, 33, 23, 13, 12]``

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
    """Build the 6 x 6 strain transformation matrix from a 3 x 3
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
