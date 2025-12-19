"""
Classes
-------
Material
IsotropicMaterial
TransverselyIsotropicMaterial
OrthotropicMaterial
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy
from numpy.typing import NDArray

from oamc.fem import utils

logger = logging.getLogger(__name__)


class Material(ABC):
    """An abstract base class for linear materials."""

    @cached_property
    @abstractmethod
    def S(self) -> NDArray:
        """Return the compliance matrix."""
        raise NotImplementedError

    @cached_property
    def C(self) -> NDArray:
        """Return the stiffness matrix."""
        return numpy.linalg.inv(self.S)

    def C_transformed(self, R: NDArray) -> NDArray:
        """Return the stiffness matrix of the material rotated by `R`.

        Parameters
        ----------
        R : numpy.ndarray
            Rotation matrix (passive convention).

        Returns
        -------
        numpy.ndarray
            Transformed stiffness matrix.
        """
        T_s = utils.T_s(R)
        T_e = utils.T_e(R)
        return T_s @ self.C @ numpy.linalg.inv(T_e)


@dataclass(frozen=True)
class IsotropicMaterial(Material):
    """A linear isotropic material.

    Attributes
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    rho : float
        Density.
    """

    E: float
    nu: float
    rho: float

    @cached_property
    def S(self) -> NDArray:
        G = self.E / (2 * (1 + self.nu))
        return numpy.array(
            [
                [1 / self.E, -self.nu / self.E, -self.nu / self.E, 0, 0, 0],
                [-self.nu / self.E, 1 / self.E, -self.nu / self.E, 0, 0, 0],
                [-self.nu / self.E, -self.nu / self.E, 1 / self.E, 0, 0, 0],
                [0, 0, 0, 1 / G, 0, 0],
                [0, 0, 0, 0, 1 / G, 0],
                [0, 0, 0, 0, 0, 1 / G],
            ]
        )


@dataclass(frozen=True)
class TransverselyIsotropicMaterial(Material):
    """A linear transversely isotropic material.

    The 1 axis is the axis of symmetry.
    The 2-3 plane is the plane of isotropy.

    Attributes
    ----------
    E1 : float
        Young's modulus.
    E2 : float
        Young's modulus.
    G12 : float
        Shear modulus.
    G23 : float
        Shear modulus.
    nu12 : float
        Poisson's ratio.
    rho : float
        Density.
    """

    E1: float
    E2: float
    nu12: float
    G23: float
    G12: float
    rho: float

    @cached_property
    def S(self) -> NDArray:
        nu23 = self.E2 / (2 * self.G23) - 1
        return numpy.array(
            [
                [1 / self.E1, -self.nu12 / self.E1, -self.nu12 / self.E1, 0, 0, 0],
                [-self.nu12 / self.E1, 1 / self.E2, -nu23 / self.E2, 0, 0, 0],
                [-self.nu12 / self.E1, -nu23 / self.E2, 1 / self.E2, 0, 0, 0],
                [0, 0, 0, 1 / self.G23, 0, 0],
                [0, 0, 0, 0, 1 / self.G12, 0],
                [0, 0, 0, 0, 0, 1 / self.G12],
            ]
        )


@dataclass(frozen=True)
class OrthotropicMaterial(Material):
    """A linear orthotropic material.

    Attributes
    ----------
    E1 : float
        Young's modulus.
    E2 : float
        Young's modulus.
    E3 : float
        Young's modulus.
    nu23 : float
        Poisson's ratio.
    nu13 : float
        Poisson's ratio.
    nu12 : float
        Poisson's ratio.
    G23 : float
        Shear modulus.
    G13 : float
        Shear modulus.
    G12 : float
        Shear modulus.
    rho : float
        Density.
    """

    E1: float
    E2: float
    E3: float
    nu23: float
    nu13: float
    nu12: float
    G23: float
    G13: float
    G12: float
    rho: float

    @cached_property
    def S(self) -> NDArray:
        return numpy.array(
            [
                [1 / self.E1, -self.nu12 / self.E1, -self.nu13 / self.E1, 0, 0, 0],
                [-self.nu12 / self.E1, 1 / self.E2, -self.nu23 / self.E2, 0, 0, 0],
                [-self.nu13 / self.E1, -self.nu23 / self.E2, 1 / self.E3, 0, 0, 0],
                [0, 0, 0, 1 / self.G23, 0, 0],
                [0, 0, 0, 0, 1 / self.G13, 0],
                [0, 0, 0, 0, 0, 1 / self.G12],
            ]
        )
