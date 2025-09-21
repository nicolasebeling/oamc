from dataclasses import dataclass
from typing import Literal

from oamc.enums import Axis


@dataclass
class BC:
    node: int
    direction: Literal[Axis.X, Axis.Y, Axis.Z]
    value: float
