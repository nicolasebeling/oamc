"""Continuum mechanics utilities for the OAMC package."""

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
