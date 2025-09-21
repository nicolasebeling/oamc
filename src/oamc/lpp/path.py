from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Path:
    coordinates: NDArray
    scalars: NDArray | None = None
