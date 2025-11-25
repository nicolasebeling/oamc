"""Constants for the OAMC package."""

import importlib.metadata

from pyvista import CellType

from oamc.enums import ElementType

VERSION = importlib.metadata.version("oamc")

BANNER = Rf"""
  ____  ___   __  ________
 / __ \/ _ | /  |/  / ___/
/ /_/ / __ |/ /|_/ / /__
\____/_/ |_/_/  /_/\___/

Optimal Additive Manufacturing of Composites
Version {VERSION}

Copyright (c) 2025 Nicolas Ebeling
MIT License
"""

NODE_COUNT_FROM_ELEMENT_TYPE = {
    ElementType.HEX8: 8,
    ElementType.HEX20: 20,
    ElementType.TET10: 10,
    ElementType.TET4: 4,
    ElementType.SURF154: 8,
    ElementType.MESH200: 8,
}


SOLID_ELEMENT_TYPES = {
    ElementType.HEX8,
    ElementType.HEX20,
    ElementType.TET10,
    ElementType.TET4,
}


SURFACE_ELEMENT_TYPES = {
    ElementType.MESH200,
}


CELL_TYPE_FROM_ELEMENT_TYPE = {
    ElementType.TET4: CellType.TETRA,
    ElementType.TET10: CellType.QUADRATIC_TETRA,
    ElementType.HEX8: CellType.HEXAHEDRON,
    ElementType.HEX20: CellType.QUADRATIC_HEXAHEDRON,
    ElementType.SURF154: CellType.QUADRATIC_QUAD,
    ElementType.MESH200: CellType.QUADRATIC_QUAD,
}

LINEAR_ELEMENT_TYPE = {
    ElementType.TET4: ElementType.TET4,
    ElementType.TET10: ElementType.TET4,
    ElementType.HEX8: ElementType.HEX8,
    ElementType.HEX20: ElementType.HEX8,
}
