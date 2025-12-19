"""Utilities for numerical optimization."""

import logging
from typing import Callable

import numpy
import scipy
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FunctionCache:
    """Cache the current function value and Jacobian matrix if its
    cheaper to compute them in parallel but an optimization algorithm
    takes them as separate arguments.
    """

    def __init__(
        self,
        fun_and_jac: Callable[[NDArray], tuple[float, NDArray]],
    ) -> None:
        self.fun_and_jac = fun_and_jac
        self._last_x = None
        self._last_fun = None
        self._last_jac = None

    def fun(self, x: NDArray) -> float:
        if self._last_x is None or not numpy.allclose(x, self._last_x):
            self._last_fun, self._last_jac = self.fun_and_jac(x)
            self._last_x = numpy.copy(x)
        return self._last_fun

    def jac(self, x: NDArray) -> NDArray:
        if self._last_x is None or not numpy.allclose(x, self._last_x):
            self._last_fun, self._last_jac = self.fun_and_jac(x)
            self._last_x = numpy.copy(x)
        return self._last_jac


def ks(
    values: NDArray,
    r: float,
    jac: NDArray | scipy.sparse.csc_array | None = None,
) -> tuple[float, NDArray]:
    """Return the Kreisselmeier-Steinhauser (KS) aggregation of the
    given values.

    Useful for for aggregating constraints or optimization objectives
    over multiple load cases, for example.

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
    sum_exponentials = numpy.sum(exponentials)
    ks_value = (max_exponent + numpy.log(sum_exponentials)) / r

    grad_ks_value = exponentials / sum_exponentials

    if jac is not None:
        grad_ks_value = grad_ks_value @ jac

    return ks_value, grad_ks_value
