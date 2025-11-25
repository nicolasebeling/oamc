"""Math utilities for the OAMC package."""

import logging

import numpy
import scipy.sparse
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


def dyadic_product(A: NDArray, B: NDArray) -> NDArray:
    return numpy.einsum("ij,kl->ijkl", A, B, optimize=True)


def bar_product(A: NDArray, B: NDArray) -> NDArray:
    return numpy.einsum("ik,jl->ijkl", A, B)


def ks(
    values: NDArray,
    r: float,
    jac: NDArray | scipy.sparse.csc_array | None = None,
) -> tuple[float, NDArray]:
    """Return the Kreisselmeier-Steinhauser (KS) aggregation of the
    given values.

    Parameters
    ----------
    values : numpy.ndarray
        Values to be aggregated.
    r : float
        KS aggregation parameter.
    jac : numpy.ndarray or scipy.sparse.csc_array, optional
        Jacobian of the given values.

    Returns
    -------
    float
        KS aggregation value.
    numpy.ndarray
        KS aggregation gradient value.

    Notes
    -----
    This is a soft (differentiable) minimum if `r < 0` and a soft
    (differentiable) maximum if `r > 0` with the following bounds:

    `min(f) + ln(n) / r <= ks(f, r) <= min(f) if r < 0`

    `max(f) <= ks(f, r) <= max(f) + ln(n) / r if r > 0`

    Hence, choose `r` such that

    `abs(r) >= ln(n) / abs(max. additive error)`

    with a magnitude between 30 and 100. Larger magnitudes of `r` yield
    a better approximation of the true minimum/maximum but may result in
    numerical instability.
    """

    if r == 0:
        raise ValueError("Aggregation parameter r must be non-zero.")

    if numpy.any(numpy.isnan(values)):
        raise ValueError("NaN in values given to KS aggregation.")

    # Apply the log-sum-exp trick to prevent overflow/underflow in exponentials:
    exponents = r * values
    max_exponent = numpy.max(exponents)
    exponentials = numpy.exp(exponents - max_exponent)
    ks_value = (max_exponent + numpy.log(numpy.sum(exponentials))) / r

    grad_ks_value = exponentials / numpy.sum(exponentials)

    if jac is not None:
        grad_ks_value = grad_ks_value @ jac

    return ks_value, grad_ks_value
