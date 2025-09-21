import logging
from typing import Callable

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OptimizationProblem:
    def __init__(
        self,
        f: Callable,
        lower_bounds: NDArray,
        upper_bounds: NDArray,
        g: list[Callable] = [],
        h: list[Callable] = [],
        # Optional first-order information:
        grad_f: Callable | None = None,
        grad_g: list[Callable | None] = [],
        grad_h: list[Callable | None] = [],
        # Optional second-order information:
        hess_f: Callable | None = None,
        hess_g: list[Callable | None] = [],
        hess_h: list[Callable | None] = [],
    ):
        self.f = f
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.g = g
        self.h = h
        self.grad_f = grad_f
        self.grad_g = grad_g
        self.grad_h = grad_h
        self.hess_f = hess_f
        self.hess_g = hess_g
        self.hess_h = hess_h
