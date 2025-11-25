from dataclasses import dataclass

from oamc.enums import Direction

ALLOWED_BC_DIRECTIONS = {Direction.X, Direction.Y, Direction.Z}


@dataclass(slots=True)
class BC:
    node: int
    direction: Direction
    value: float

    def __post_init__(self):
        if self.node < 0:
            raise ValueError("Node ID must be non-negative.")
        if self.direction not in ALLOWED_BC_DIRECTIONS:
            raise ValueError(f"Direction must be in {ALLOWED_BC_DIRECTIONS}.")
