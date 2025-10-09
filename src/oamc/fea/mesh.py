import logging
from dataclasses import dataclass
from functools import cached_property
from time import perf_counter as timer

import numpy
from numpy.typing import NDArray
from scipy.optimize import root
from scipy.spatial import ConvexHull, KDTree

from oamc.enums import ElementType
from oamc.fea import utils

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mesh:
    """A finite-element mesh.

    :param nodes: nodal coordinates as an array of shape (number of nodes, 3)
    :param element_connectivity: element connectivity array of shape (number of
        elements, number of nodes per element)
    :param element_type: element type
    """

    nodes: NDArray[numpy.float64]
    element_type: ElementType
    element_connectivity: NDArray[numpy.int32]

    def n_dof(self) -> int:
        return self.nodes.shape[0] * 3

    def n_elements(self) -> int:
        return self.element_connectivity.shape[0]

    @cached_property
    def centroids(self) -> NDArray:
        start = timer()
        centroids = self.nodes[self.element_connectivity].mean(axis=1)
        logger.info(f"Centroids computed in {round(timer() - start, 3)} seconds.")
        return centroids

    @cached_property
    def tree(self) -> KDTree:
        start = timer()
        tree = KDTree(self.centroids)
        logger.info(f"Search tree computed in {round(timer() - start, 3)} seconds.")
        return tree

    @cached_property
    def hulls(self) -> list[ConvexHull]:
        start = timer()
        hulls = [ConvexHull(self.nodes[nodes]) for nodes in self.element_connectivity]
        logger.info(f"Convex hulls computed in {round(timer() - start, 3)} seconds.")
        return hulls

    @cached_property
    def volume(self) -> float:
        volume = 0
        for hull in self.hulls:
            volume += hull.volume
        return volume

    @cached_property
    def _precompute(self) -> tuple[NDArray, NDArray]:
        """
        :return dN_dxyz: shape function gradients at all integration points of
            all elements in the mesh
        :return w_det_dxyz_drst: product of the integration weight and the
            determinant of the Jacobian at all integration points of all
            elements in the mesh
        """
        start = timer()

        points = utils.INT_POINTS[self.element_type]
        weights = utils.INT_WEIGHTS[self.element_type]

        # Shape function derivatives with respect to natural coordinates at all
        # integration points:

        dN_drst = numpy.empty(
            shape=(
                len(points),
                self.element_connectivity.shape[1],
                3,
            ),
        )

        for point_index, point in enumerate(points):
            dN_drst[point_index] = utils.dN_drst(
                element_type=self.element_type,
                rst=point,
            )

        # Shape function derivatives with respect to global coordinates and
        # integration factors at all integration points of all elements:

        dN_dxyz = numpy.empty(
            shape=(
                self.element_connectivity.shape[0],
                len(points),
                self.element_connectivity.shape[1],
                3,
            ),
        )

        w_det_dxyz_drst = numpy.empty(
            shape=(
                self.element_connectivity.shape[0],
                len(points),
            )
        )

        for element_index, node_indices in enumerate(self.element_connectivity):
            for point_index, (point, weight) in enumerate(zip(points, weights)):
                dxyz_drst = self.nodes[node_indices].T @ dN_drst[point_index]
                dN_dxyz[element_index, point_index] = dN_drst[point_index] @ numpy.linalg.inv(
                    dxyz_drst
                )
                w_det_dxyz_drst[element_index, point_index] = weight * numpy.linalg.det(dxyz_drst)

        logger.info(
            f"Geometric factors precomputed and cached in {round(timer() - start, 3)} seconds."
        )

        return dN_dxyz, w_det_dxyz_drst

    @property
    def dN_dxyz(self) -> NDArray:
        """
        :return dN_dxyz: shape function gradients at all integration points of
            all elements in the mesh
        """
        return self._precompute[0]

    @property
    def w_det_dxyz_drst(self) -> NDArray:
        """
        :return w_det_dxyz_drst: product of the integration weight and the
            determinant of the Jacobian at all integration points of all elements in the mesh
        """
        return self._precompute[1]

    def element_contains(self, element: int, x: NDArray, tolerance: float = 1e-9) -> bool:
        """Check if the given element contains the given point.

        :param element: index of the element
        :param x: point to check
        :param tolerance: tolerance to account for numerical inaccuracies
        :return: whether the point is contained in the element
        """
        hull = self.hulls[element]
        return all(hull.equations[:, 0:3] @ x + hull.equations[:, 3] <= tolerance)

    def xyz(self, element: int, rst: NDArray) -> NDArray:
        """
        :param element: index of the element containing the point
        :param rst: point in local natural coordinates (xi, eta, zeta) in [-1, 1]
        :return: point in global coordinates (x, y, z)
        """
        # Do not check the validity of the natural coordinates as the root
        # finding algorithm in Mesh.rst may exceed them.
        return utils.N(self.element_type, rst).T @ self.nodes[self.element_connectivity[element]]

    def rst(self, xyz: NDArray, k: int = 27) -> tuple[int, NDArray]:
        """
        :param xyz: point in global cartesian coordinates (x, y, z)
        :param k: number of candidate elements to query from the k-d tree
        :return: index of the element containing the point, point in natural
            coordinates (xi, eta, zeta) of that element
        :raises ValueError: if the point is not inside the part or number of
            candidate elements queried from the k-d tree is too low
        """
        candidates = self.tree.query(xyz, k)[1]
        for candidate in candidates:
            if self.element_contains(candidate, xyz):
                rst = root(
                    fun=lambda x: self.xyz(candidate, x) - xyz,
                    x0=numpy.zeros(3),
                    options={"xtol": 1e-9},
                ).x
                return candidate, rst
        raise ValueError(
            "The point is not inside the part or the number of candidate elements queried from the"
            "k-d tree is too low."
        )
