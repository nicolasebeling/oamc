import logging
from typing import Callable

import numpy
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FunctionCache:
    def __init__(
        self,
        name: str,
        fun_and_jac: Callable[[NDArray], tuple[float, NDArray]],
    ) -> None:
        self.name = name
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
