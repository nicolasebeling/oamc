from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class OptimizationResult:
    x: NDArray
    f: float
