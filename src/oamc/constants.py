"""Constants for the OAMC package."""

import importlib.metadata

from pyvista import CellType

from oamc.enums import ElementType

VERSION = importlib.metadata.version("oamc")

BANNER = rf"""
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
}


CELL_TYPE_FROM_ELEMENT_TYPE = {
    ElementType.HEX8: CellType.HEXAHEDRON,
    ElementType.HEX20: CellType.QUADRATIC_HEXAHEDRON,
    ElementType.TET10: CellType.QUADRATIC_TETRA,
    ElementType.TET4: CellType.TETRA,
}
