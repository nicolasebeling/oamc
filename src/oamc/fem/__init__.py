"""Classes and functions related to the finite-element method."""

from .bc import BC
from .material import (
    IsotropicMaterial,
    Material,
    OrthotropicMaterial,
    TransverselyIsotropicMaterial,
)
from .mesh import Mesh, SolidMesh, SurfaceMesh
from .model import SolidModel
