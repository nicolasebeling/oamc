"""Enums for the OAMC package."""

from enum import Enum, auto


class Direction(Enum):
    # Cartesian axes:
    X = 0
    Y = 1
    Z = 2

    # Principal axes:
    P1 = 3
    P2 = 4
    P3 = 5

    # Principal axes aliases:
    MAX = P1
    INT = P2
    MIN = P3

    # Special directions:
    NORMAL = 6


class ElementType(Enum):
    """Finite element types."""

    # General 3D element types:
    TET4 = "TET4"
    TET10 = "TET10"
    HEX8 = "HEX8"
    HEX20 = "HEX20"

    # Corresponding Ansys 3D element types:
    SOLID185 = HEX8
    SOLID186 = HEX20
    SOLID187 = TET10
    SOLID285 = TET4

    # Special Ansys 2D element type:
    SURF154 = "SURF154"
    MESH200 = "MESH200"


class ProjectionMethod(Enum):
    """Methods for projecting values from integration points to nodes."""

    L2 = auto()
    ANSYS = auto()


class RKMethod(Enum):
    """Runge-Kutta (RK) integration schemes."""

    RK4 = auto()
