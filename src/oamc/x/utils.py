import logging
from typing import Callable, Iterable, Literal
import numpy
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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


def ks(
    f: NDArray,
    r: float,
    compute_derivatives: bool = False,
    jac_f: NDArray | None = None,
) -> float | tuple[float, NDArray]:
    """
    Return the Kreisselmeier-Steinhauser (KS) aggregation of an array of
    values or an iterable of scalar-valued functions.

    This is a soft (differentiable) minimum if `r < 0` and a soft
    maximum if `r > 0` with the following bounds:

    `max(f) <= ks(f, r) <= max(f) + ln(n) / r`

    Hence, choose `r` such that `r >= ln(n) / max. additive error` with
    a magnitude between 30 and 100. Larger magnitudes of `r` yield a
    better approximation of the true minimum/maximum but may result in
    numerical instability.

    :param f: function values
    :param r: aggregation parameter
    :param jac_f: gradients of the functions with respect to some parameter
        vector p in standard jacobian format (optional)
    :return: KS aggregation of function values, gradient of the KS aggregation
    """

    if r == 0:
        raise ValueError("Aggregation parameter r must be non-zero.")

    # Apply the log-sum-exp trick to prevent overflow/underflow in exponentials:
    exponents = r * f
    max_exponent = numpy.max(exponents)
    exponentials = numpy.exp(exponents - max_exponent)
    ks_value = (max_exponent + numpy.log(numpy.sum(exponentials))) / r

    if not compute_derivatives:
        return ks_value

    if jac_f is None:
        grad_ks_value = exponentials / numpy.sum(exponentials)
    else:
        grad_ks_value = jac_f.T @ exponentials / numpy.sum(exponentials)

    return ks_value, grad_ks_value
