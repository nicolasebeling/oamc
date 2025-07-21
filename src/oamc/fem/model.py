from dataclasses import dataclass
from functools import cached_property
from time import perf_counter as timer

import numpy
from numpy.typing import NDArray
from scipy.optimize import root
from scipy.spatial import ConvexHull, KDTree

from oamc.fem.ansys_element_type import AnsysElementType
from oamc.fem.functions import node_count, shape_functions
from oamc.mechanics.functions import convert_to_tensor


@dataclass
class Model:
    nodes: NDArray
    elements: NDArray
    types: list[AnsysElementType]
    stresses: NDArray

    @cached_property
    def element_tree(self) -> KDTree:
        start = timer()
        tree = KDTree([self.element_centroid(element) for element in range(len(self.elements))])
        print(f"Search tree computed in {round(timer() - start, 3)} seconds.")
        return tree

    @cached_property
    def element_hulls(self) -> list[ConvexHull]:
        start = timer()
        hulls = [ConvexHull(self.nodes[self.elements[i]]) for i in range(len(self.elements))]
        print(f"Convex hulls computed in {round(timer() - start, 3)} seconds.")
        return hulls

    def element_centroid(self, element: int) -> NDArray:
        """
        Computes the centroid of the given element.

        :param element: index of the element
        :return: centroid of the element
        """
        return self.nodes[self.elements[element]].mean(axis=0)

    def element_contains(self, element: int, point: NDArray, tolerance: float = 1e-9) -> bool:
        """
        Checks if the given point is contained in the element.

        :param element: index of the element
        :param point: point to check
        :param tolerance: tolerance to account for numerical inaccuracies
        :return: `True` if the point is contained in the element, `False` otherwise
        """
        hull = self.element_hulls[element]
        return all(hull.equations[:, 0:3] @ point + hull.equations[:, 3] <= tolerance)

    def global_coordinates(self, element: int, x: NDArray) -> NDArray:
        """
        :param x: point in natural coordinates (xi, eta, zeta) in [-1, 1]
        :return: point in global coordinates (x, y, z)
        """
        # Do not check validity of natural coordinates as the root finding algorithm in Model.natural_coordinates may exceed them.
        # if any(x < -1) or any(x > 1):
        #     raise ValueError("Natural coordinates (xi, eta, zeta) must be in [-1, 1].")
        type = self.types[element]
        nodes = self.elements[element, : node_count(type)]
        N = shape_functions(type, x)
        return N.T @ self.nodes[nodes]

    def natural_coordinates(self, point: NDArray, k: int = 27) -> tuple[int, NDArray]:
        """
        :param point: point in global coordinates (x, y, z)
        :param k: number of candidate elements to query from the k-d tree
        :return: index of the element containing the point
        :return: point in natural coordinates (xi, eta, zeta) of that element
        """
        candidates = self.element_tree.query(point, k)[1]
        for element in candidates:
            if self.element_contains(element, point):
                x = root(
                    fun=lambda x: self.global_coordinates(element, x) - point,
                    x0=numpy.zeros(3),
                    options={"xtol": 1e-9},
                ).x
                return element, x
        raise ValueError(
            "The point is not inside the part or the mesh is distorted. "
            "In the latter case, increase the number of candidate elements "
            "queried from the k-d tree."
        )

    def stress(self, point: NDArray, k: int = 27) -> NDArray:
        """
        Interpolates the stress at `point` from stresses at nodes.

        NOTE: Stresses at nodes are extrapolated from stresses at Gauss points.
        Directly extrapolating from Gauss points may be more accurate but less
        smooth than interpolating from nodes using the element's shape function.
        As elements are assumed to be small and fiber paths shall be as smooth
        as possible, stresses are interpolated from nodes for this application.

        :param point: point where the stress shall be computed
        :param k: number of candidate elements to query from the k-d tree
        :return: stress tensor at `point`
        """
        element, natural_coordinates = self.natural_coordinates(point, k)
        type = self.types[element]
        N = shape_functions(type, natural_coordinates)
        stresses = self.stresses[self.elements[element, : node_count(type)]]
        return convert_to_tensor(numpy.average(stresses, axis=0, weights=N))
