"""Math utilities."""

import logging

import numpy
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def skew(v: NDArray) -> NDArray:
    """Return the skew-symmetric matrix of a 3D vector.

    Parameters
    ----------
    v : numpy.ndarray
        A 3D vector.

    Returns
    -------
    numpy.ndarray
        The skew-symmetric matrix of the 3D vector.
    """

    if v.shape != (3,):
        raise ValueError(f"Vector must have shape (3,) but has shape {v.shape}.")

    return numpy.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=float,
    )


def tensor_product(A: NDArray, B: NDArray) -> NDArray:
    return numpy.tensordot(A, B, axes=0)
