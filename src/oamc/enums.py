"""Enums for the OAMC package."""

from enum import Enum, auto


class Axis(Enum):
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


class ElementType(Enum):
    # General 3D element types:
    HEX8 = auto()
    HEX20 = auto()
    TET10 = auto()
    TET4 = auto()

    # Corresponding Ansys element types:
    SOLID185 = HEX8
    SOLID186 = HEX20
    SOLID187 = TET10
    SOLID285 = TET4


class RKMethod(Enum):
    RK4 = auto()
