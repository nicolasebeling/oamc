from typing import Literal
import numpy
from numpy.typing import NDArray


def skew(v: NDArray) -> NDArray:
    """Return the skew-symmetric matrix of a 3D vector.

    :param v: a 3D vector
    :return: the skew-symmetric matrix of v
    """
    return numpy.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=numpy.float64,
    )


def dyadic_product(A: NDArray, B: NDArray) -> NDArray:
    return numpy.einsum("ij,kl->ijkl", A, B, optimize=True)


def bar_product(A: NDArray, B: NDArray) -> NDArray:
    return numpy.einsum("ik,jl->ijkl", A, B)


def tensor_to_matrix(
    tensor: NDArray,
    convention: Literal["voigt", "engineering", "mandel"] = "engineering",
) -> NDArray:
    """
    :param A: fourth-order tensor of shape (3, 3, 3, 3)
    :return: 6 x 6 matrix in engineering Voigt notation
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
            dtype=numpy.float64,
        )
        return matrix / factor[None, :]

    if convention.lower() == "mandel":
        factor = numpy.array(
            [1, 1, 1, numpy.sqrt(2), numpy.sqrt(2), numpy.sqrt(2)],
            dtype=numpy.float64,
        )
        return factor[:, None] * matrix / factor[None, :]

    raise ValueError("Unsupported convention.")
