"""
Classes
-------
Mesh
SurfaceMesh
SolidMesh
"""

import logging
from dataclasses import dataclass
from functools import cached_property
from time import perf_counter as timer

import numpy
import pyvista
from numpy.typing import NDArray
from scipy.optimize import root
from scipy.spatial import ConvexHull, KDTree

from oamc.enums import ElementType
from oamc.fem import utils
from oamc.utils.vtk import convert_to_triangle_mesh

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mesh:
    """A finite-element mesh.

    Attributes
    ----------
    nodes : NDArray
        Nodal coordinates as an array of shape (number of nodes, 3).
    type : oamc.enums.ElementType
        Element type.
    connectivity : NDArray
        Element connectivity as an array of shape (number of elements, number of nodes per element).
    """

    nodes: NDArray[numpy.float64]
    type: ElementType
    connectivity: NDArray[numpy.int32]

    @cached_property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @cached_property
    def n_int_points(self) -> int:
        return utils.N_INT_POINTS[self.type]

    @cached_property
    def n_elements(self) -> int:
        return self.connectivity.shape[0]

    @cached_property
    def n_dofs(self) -> int:
        return self.n_nodes * 3

    @cached_property
    def centroids(self) -> NDArray:
        start = timer()
        # TODO: This is only correct for linear 1D elements, triangles, and tetrahedra.
        centroids = self.nodes[self.connectivity].mean(axis=1)
        logger.info(f"Centroids computed in {round(timer() - start, 3)} seconds.")
        return centroids

    @cached_property
    def tree(self) -> KDTree:
        start = timer()
        tree = KDTree(self.centroids)
        logger.info(f"Search tree computed in {round(timer() - start, 3)} seconds.")
        return tree


@dataclass(frozen=True)
class SurfaceMesh(Mesh):
    """A finite-element surface mesh.

    Attributes
    ----------
    nodes : NDArray
        Nodal coordinates as an array of shape (number of nodes, 3).
    type : oamc.enums.ElementType
        Element type.
    connectivity : NDArray
        Element connectivity as an array of shape (number of elements, number of nodes per element).
    """

    @cached_property
    def polydata(self) -> pyvista.PolyData:
        faces = numpy.concatenate([numpy.insert(face[:4], 0, 4) for face in self.connectivity])
        return convert_to_triangle_mesh(pyvista.PolyData(self.nodes, faces))

    def get_closest_points(self, points: NDArray) -> NDArray:
        return self.polydata.find_closest_cell(
            points,
            return_closest_point=True,
        )[1]

    def get_unit_normal_vectors(self, points: NDArray) -> NDArray:
        vectors = points - self.get_closest_points(points)
        vectors /= numpy.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    def get_distances(self, points: NDArray) -> NDArray:
        vectors = points - self.get_closest_points(points)
        return numpy.linalg.norm(vectors, axis=vectors.ndim - 1)


@dataclass(frozen=True)
class SolidMesh(Mesh):
    """A finite-element solid mesh.

    Attributes
    ----------
    nodes : NDArray
        Nodal coordinates as an array of shape (number of nodes, 3).
    type : oamc.enums.ElementType
        Element type.
    connectivity : NDArray
        Element connectivity as an array of shape (number of elements,
        number of nodes per element).
    """

    @cached_property
    def hulls(self) -> list[ConvexHull]:
        start = timer()
        hulls = [ConvexHull(self.nodes[nodes]) for nodes in self.connectivity]
        logger.info(f"Convex hulls computed in {round(timer() - start, 3)} seconds.")
        return hulls

    @cached_property
    def volume(self) -> float:
        volume = 0
        for hull in self.hulls:
            volume += hull.volume
        return volume

    @cached_property
    def N(self) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Shape function values at all integration points.
        """
        points = utils.INT_POINTS[self.type]
        N = numpy.empty(
            shape=(
                len(points),
                self.connectivity.shape[1],
            ),
        )
        for point_index, point in enumerate(points):
            N[point_index] = utils.N(
                element_type=self.type,
                rst=point,
            )
        return N

    @cached_property
    def dN_drst(self) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Shape function derivatives with respect to natural
            coordinates at all integration points.
        """
        points = utils.INT_POINTS[self.type]
        dN_drst = numpy.empty(
            shape=(
                len(points),
                self.connectivity.shape[1],
                3,
            ),
        )
        for point_index, point in enumerate(points):
            dN_drst[point_index] = utils.dN_drst(
                element_type=self.type,
                rst=point,
            )
        return dN_drst

    @cached_property
    def _precompute_dN_dxyz_and_w_det_dxyz_drst(self) -> tuple[NDArray, NDArray]:
        start = timer()

        weights = utils.INT_WEIGHTS[self.type]

        # Shape function derivatives with respect to global coordinates and
        # integration factors at all integration points of all elements:

        dN_dxyz = numpy.empty(
            shape=(
                self.n_elements,
                len(weights),
                self.connectivity.shape[1],
                3,
            ),
        )

        w_det_dxyz_drst = numpy.empty(
            shape=(
                self.n_elements,
                len(weights),
            )
        )

        for element_index, node_indices in enumerate(self.connectivity):
            for point_index, weight in enumerate(weights):
                dxyz_drst = self.nodes[node_indices].T @ self.dN_drst[point_index]
                dN_dxyz[element_index, point_index] = self.dN_drst[point_index] @ numpy.linalg.inv(
                    dxyz_drst
                )
                det_dxyz_drst = numpy.linalg.det(dxyz_drst)
                if det_dxyz_drst <= 0:
                    logger.warning("Negative Jacobian determinant.")
                w_det_dxyz_drst[element_index, point_index] = weight * det_dxyz_drst

        logger.info(
            f"Geometric factors precomputed and cached in {round(timer() - start, 3)} seconds."
        )

        return dN_dxyz, w_det_dxyz_drst

    @property
    def dN_dxyz(self) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Shape function gradients at all integration points of all
            elements in the mesh as an array of shape (number of
            elements, number of integration points per element, number
            of nodes per element, 3).
        """
        return self._precompute_dN_dxyz_and_w_det_dxyz_drst[0]

    @property
    def w_det_dxyz_drst(self) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Product of the integration weight and the determinant of the
            Jacobian at all integration points of all elements in the
            mesh as an array of shape (number of elements, number of
            integration points per element).
        """
        return self._precompute_dN_dxyz_and_w_det_dxyz_drst[1]

    def element_contains(self, element: int, x: NDArray, tolerance: float = 1e-9) -> bool:
        """Check if the given element contains the given point.

        :param element: index of the element
        :param x: point to check
        :param tolerance: tolerance to account for numerical inaccuracies
        :return: whether the point is contained in the element
        """
        hull = self.hulls[element]
        return all(hull.equations[:, 0:3] @ x + hull.equations[:, 3] <= tolerance)

    def get_xyz(self, element: int, rst: NDArray) -> NDArray:
        """
        :param element: index of the element containing the point
        :param rst: point in local natural coordinates (xi, eta, zeta) in [-1, 1]
        :return: point in global coordinates (x, y, z)
        """
        # Do not check the validity of the natural coordinates as the root
        # finding algorithm in Mesh.rst may exceed them.
        return utils.N(self.type, rst).T @ self.nodes[self.connectivity[element]]

    def get_rst(self, xyz: NDArray, k: int = 27) -> tuple[int, NDArray]:
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
                    fun=lambda x: self.get_xyz(candidate, x) - xyz,
                    x0=numpy.zeros(3),
                    options={"xtol": 1e-9},
                ).x
                return candidate, rst
        raise ValueError(
            "The point is not inside the part or the number of candidate "
            "elements queried from the k-d tree is too low."
        )
