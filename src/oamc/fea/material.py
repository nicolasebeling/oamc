import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy
from numpy.typing import NDArray

from oamc.fea import utils

logger = logging.getLogger(__name__)


class Material(ABC):
    """An abstract base class for linear materials."""

    @cached_property
    @abstractmethod
    def S(self) -> NDArray:
        raise NotImplementedError

    @cached_property
    def C(self) -> NDArray:
        return numpy.linalg.inv(self.S)

    def C_transformed(self, R: NDArray) -> NDArray:
        """
        :param R: rotation matrix (active convention)
        :return: transformed stiffness matrix
        """
        T_s = utils.T_s(R)
        T_e = utils.T_e(R)
        return T_s @ self.C @ numpy.linalg.inv(T_e)


@dataclass(frozen=True)
class IsotropicMaterial(Material):
    """A linear isotropic material.

    :param E: Young's modulus
    :param nu: Poisson's ratio
    :param rho: density
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

    In fiber composite materials, 1 is the fiber direction.

    :param E1: Young's modulus (in fiber direction)
    :param E2: Young's modulus (in transverse direction)
    :param G12: shear modulus
    :param G23: shear modulus
    :param nu12: Poisson's ratio
    :param rho: density
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

    In fiber composite (transverse isotropic) materials, 1 is the fiber direction.

    :param E1: Young's modulus (in fiber direction)
    :param E2: Young's modulus
    :param E3: Young's modulus
    :param nu12: Poisson's ratio
    :param nu23: Poisson's ratio
    :param nu13: Poisson's ratio
    :param G23: shear modulus
    :param G13: shear modulus
    :param G12: shear modulus
    :param rho: density
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
