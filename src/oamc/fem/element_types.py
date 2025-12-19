"""Currently not in use."""

from dataclasses import dataclass
from itertools import product

import numpy
from pyvista import CellType

# TODO: Consider integrating new ElementType class in existing code.


@dataclass(frozen=True, slots=True)
class ElementType:
    cell_type: CellType
    node_count: int
    int_points: tuple[tuple[float, float, float], ...]
    int_weights: tuple[float, ...]

    def __post_init__(self):
        if len(self.int_points) != len(self.int_weights):
            raise ValueError(
                "The number of integration weights must equal the number of integration points."
            )


HEX8 = ElementType(
    cell_type=CellType.HEXAHEDRON,
    node_count=8,
    int_points=tuple(product((-numpy.sqrt(1 / 3), numpy.sqrt(1 / 3)), repeat=3)),
    int_weights=tuple([wx * wy * wz for wx, wy, wz in product((1, 1), repeat=3)]),
)

HEX20 = ElementType(
    cell_type=CellType.QUADRATIC_HEXAHEDRON,
    node_count=20,
    int_points=tuple(product((-numpy.sqrt(3 / 5), 0, numpy.sqrt(3 / 5)), repeat=3)),
    int_weights=tuple([wx * wy * wz for wx, wy, wz in product((5 / 9, 8 / 9, 5 / 9), repeat=3)]),
)

TET10 = ElementType(
    cell_type=CellType.QUADRATIC_TETRA,
    node_count=10,
    int_points=(
        (0.13819660, 0.13819660, 0.13819660),
        (0.58541020, 0.13819660, 0.13819660),
        (0.13819660, 0.58541020, 0.13819660),
        (0.13819660, 0.13819660, 0.58541020),
    ),
    int_weights=(
        1 / 24,
        1 / 24,
        1 / 24,
        1 / 24,
    ),
)

TET4 = ElementType(
    cell_type=CellType.TETRA,
    node_count=4,
    int_points=((0.25, 0.25, 0.25),),
    int_weights=(1 / 6,),
)
