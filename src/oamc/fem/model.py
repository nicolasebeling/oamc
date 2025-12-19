"""
Classes
-------
SolidModel
"""

import logging
from copy import deepcopy
from functools import cached_property
from time import perf_counter as clock
from typing import Generic, TypeVar

import numpy
import pypardiso
import pyvista
import scipy.sparse
import scipy.sparse.linalg

# from numba import float64, njit
from numpy.typing import NDArray

from oamc.constants import CELL_TYPE_FROM_ELEMENT_TYPE, NODE_COUNT_FROM_ELEMENT_TYPE
from oamc.enums import ElementType, ProjectionMethod
from oamc.fem import utils
from oamc.fem.bc import BC
from oamc.fem.material import Material
from oamc.fem.mesh import SolidMesh

logger = logging.getLogger(__name__)

# TODO: Implement element stiffness matrix calculations in Numba for speedup.

# @njit(
#     (float64[:, :])(float64[:, :], float64[:, :]),
#     cache=True,
#     fastmath=True,
# )
# def K_HEX8_COO(X, C):
#     """
#     :param X: nodal coordinates
#     :param C: material stiffness matrix
#     :return: element stiffness matrix
#     """
#     ...
#     return numpy.zeros((6, 6), dtype=float)


# @njit(
#     (float64[:, :])(float64[:, :], float64[:, :]),
#     cache=True,
#     fastmath=True,
# )
# def K_HEX20_COO(X, C):
#     """
#     :param X: nodal coordinates
#     :param C: material stiffness matrix
#     :return: element stiffness matrix
#     """
#     return numpy.zeros((6, 6), dtype=float)


# @njit(
#     (float64[:, :])(float64[:, :], float64[:, :]),
#     cache=True,
#     fastmath=True,
# )
# def K_TET10_COO(X, C):
#     """
#     :param X: nodal coordinates
#     :param C: material stiffness matrix
#     :return: element stiffness matrix
#     """
#     return numpy.zeros((6, 6), dtype=float)


# @njit(
#     (float64[:, :])(float64[:, :], float64[:, :]),
#     cache=True,
#     fastmath=True,
# )
# def K_TET4_COO(X, C):
#     """
#     :param X: nodal coordinates
#     :param C: material stiffness matrix
#     :return: element stiffness matrix
#     """
#     return numpy.zeros((6, 6), dtype=float)

# As Material has various subclasses:
MATERIAL = TypeVar("MATERIAL", bound=Material)


class SolidModel(Generic[MATERIAL]):
    """A linear static finite-element model with 3D elements."""

    def __init__(
        self,
        mesh: SolidMesh,
        material: MATERIAL,
        dbc: list[BC],
        nbc: list[BC],
    ) -> None:
        """
        Parameters
        ----------
        mesh : oamc.fem.mesh.SolidMesh
            Solid finite-element mesh.
        material : oamc.fem.material.Material
            Material model.
        dbc : list of oamc.fem.bc.BC
            Dirichlet (essential) boundary conditions.
        nbc : list of oamc.fem.bc.BC
            Neumann (natural) boundary conditions.
        """
        self.mesh = mesh
        self.material = material
        self.dbc = dbc
        self.nbc = nbc

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "C" not in cls.__dict__:
            logger.warning(
                "Computation of local material stiffness matrix not overridden by subclass."
            )

    def __setattr__(self, name, value) -> None:
        if name in self._DEP_MAP:
            for cached_value in self._DEP_MAP[name]:
                self.__dict__.pop(cached_value, None)
        super().__setattr__(name, value)

    _DEP_MAP = {
        "mesh": {
            "K",
            "f",
            "u",
            "_precompute_strain_and_stress_by_L2_projection",
            "_precompute_strain_and_stress_by_extrapolation",
        },
        "material": {
            "K",
            "u",
            "_precompute_strain_and_stress_by_L2_projection",
            "_precompute_strain_and_stress_by_extrapolation",
        },
        "dbc": {
            "u",
            "_precompute_strain_and_stress_by_L2_projection",
            "_precompute_strain_and_stress_by_extrapolation",
        },
        "nbc": {
            "f",
            "u",
            "_precompute_strain_and_stress_by_L2_projection",
            "_precompute_strain_and_stress_by_extrapolation",
        },
    }

    @classmethod
    def _merge_dependencies(
        cls,
        dependencies: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        merged = deepcopy(cls._DEP_MAP)
        for key, value in dependencies.items():
            merged.setdefault(key, set()).update(value)
        return merged

    def get_grid(
        self,
        u_scaling_factor: float = 0,
    ) -> pyvista.UnstructuredGrid:
        """
        Parameters
        ----------
        u_scaling_factor : float, default: 0
            Scaling factor for the deformation. 0 means no deformation.

        Returns
        -------
        pyvista.UnstructuredGrid
            The (deformed) mesh as an unstructured grid.
        """

        # Each cell (finite element in this context) must start with the
        # number of points followed by the indices of its nodes:
        node_count_per_element = NODE_COUNT_FROM_ELEMENT_TYPE[self.mesh.type]
        cells = numpy.column_stack(
            (
                numpy.full(self.mesh.n_elements, node_count_per_element),
                self.mesh.connectivity[:, :node_count_per_element],
            )
        )
        types = [CELL_TYPE_FROM_ELEMENT_TYPE[self.mesh.type]] * len(cells)

        nodes = self.mesh.nodes.copy()
        if u_scaling_factor != 0:
            nodes += self.u.reshape(-1, 3) * u_scaling_factor

        return pyvista.UnstructuredGrid(
            cells,
            types,
            nodes,
        )

    def C(
        self,
        element_index: int,
        int_point_index: int,
    ) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Material stiffness matrix of shape (6, 6).
        """
        return self.material.C

    @cached_property
    def K(self) -> scipy.sparse.csr_array:
        """
        Returns
        -------
        scipy.sparse.csr_array
            Global stiffness matrix.
        """

        start = clock()

        rows, columns, values = [], [], []

        for element_index, node_indices in enumerate(self.mesh.connectivity):
            # Determine degrees of freedom per element:
            dofs = utils.dof_indices(node_indices)

            # Initialize element stiffness matrix:
            K_e = numpy.zeros((dofs.size, dofs.size))

            # Compute element stiffness matrix:
            for int_point_index, (jac_N, w_det_J) in enumerate(
                zip(
                    self.mesh.dN_dxyz[element_index],
                    self.mesh.w_det_dxyz_drst[element_index],
                )
            ):
                B = utils.B(jac_N)
                K_e += (B.T @ self.C(element_index, int_point_index) @ B) * w_det_J

            # Add to global stiffness matrix:
            rows.append(numpy.repeat(dofs, dofs.size))
            columns.append(numpy.tile(dofs, dofs.size))
            values.append(K_e.ravel())

        # Assemble the global stiffness matrix in COO format, then
        # convert to CSR format for solving:
        K = scipy.sparse.coo_array(
            arg1=(
                numpy.concatenate(values),
                (
                    numpy.concatenate(rows),
                    numpy.concatenate(columns),
                ),
            ),
            shape=(self.mesh.n_dofs, self.mesh.n_dofs),
        ).tocsr()

        K.sort_indices()

        logger.info(f"Global stiffness matrix assembled in {round(clock() - start, 3)} seconds.")

        return K

    @cached_property
    def f(self) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Global force vector.
        """
        start = clock()

        f = numpy.zeros(self.mesh.nodes.shape[0] * 3)

        # Apply Neumann boundary conditions:
        for nbc in self.nbc:
            f[nbc.node * 3 + nbc.direction.value] += nbc.value

        logger.info(f"Global force vector assembled in {round(clock() - start, 3)} seconds.")
        logger.debug(f"Resultant of the applied load: {numpy.sum(f.reshape(-1, 3), axis=0)}")

        return f

    @cached_property
    def u(self) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Nodal displacement vector.
        """

        start = clock()

        n = self.mesh.nodes.shape[0] * 3

        # Apply Dirichlet boundary conditions:
        u_dbc = numpy.zeros(n)
        constrained = numpy.zeros(n, dtype=bool)
        for dbc in self.dbc:
            dof_index = dbc.node * 3 + dbc.direction.value
            constrained[dof_index] = True
            u_dbc[dof_index] = dbc.value

        free = ~constrained
        K_ff = self.K[free][:, free]
        K_fc = self.K[free][:, constrained]
        f_f = self.f[free] - K_fc @ u_dbc[constrained]

        logger.info(f"Boundary conditions applied in {round(clock() - start, 3)} seconds.")

        start = clock()

        u_f = pypardiso.spsolve(K_ff, f_f)

        # Alternative solvers:

        # ilu = scipy.sparse.linalg.spilu(K_ff)
        # preconditioner = scipy.sparse.linalg.LinearOperator(K_ff.shape, ilu.solve)
        # u_f, info = scipy.sparse.linalg.cg(
        #     A=K_ff,
        #     b=f_f,
        #     atol=1e-10,
        #     maxiter=2000,
        #     M=preconditioner,
        # )
        # logger.info(f"SciPy CG solver exit code: {info}")

        # u_f = scipy.sparse.linalg.spsolve(K_ff, f_f)

        u = numpy.zeros(n)
        u[free] = u_f
        u[constrained] = u_dbc[constrained]

        logger.info(f"System solved in {round(clock() - start, 3)} seconds.")

        return u

    @cached_property
    def _precompute_strain_and_stress_by_L2_projection(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Compute strain and stress values at quadrature points and
        extrapolate them to nodes by mass-lumped L2 projection.
        """
        start = clock()

        int_point_e = numpy.empty((self.mesh.n_elements, self.mesh.n_int_points, 6))
        int_point_s = numpy.empty((self.mesh.n_elements, self.mesh.n_int_points, 6))

        # Accumulators for nodal projection:
        nodal_sum_of_weights = numpy.zeros(self.mesh.n_nodes)
        nodal_sum_of_weighted_e = numpy.zeros((self.mesh.n_nodes, 6))
        nodal_sum_of_weighted_s = numpy.zeros((self.mesh.n_nodes, 6))

        N_at_int_points = numpy.array(
            [utils.N(self.mesh.type, point) for point in utils.INT_POINTS[self.mesh.type]]
        )

        for element_index, node_indices in enumerate(self.mesh.connectivity):
            dofs = utils.dof_indices(node_indices)

            for int_point_index, (N_values, dN_dxyz, w_det_J) in enumerate(
                zip(
                    N_at_int_points,
                    self.mesh.dN_dxyz[element_index],
                    self.mesh.w_det_dxyz_drst[element_index],
                )
            ):
                int_point_e[element_index, int_point_index] = utils.B(dN_dxyz) @ self.u[dofs]
                int_point_s[element_index, int_point_index] = (
                    self.C(element_index, int_point_index)
                    @ int_point_e[element_index, int_point_index]
                )

                nodal_weights = N_values * w_det_J
                nodal_sum_of_weights[node_indices] += nodal_weights
                nodal_sum_of_weighted_e[node_indices] += (
                    nodal_weights[:, None] * int_point_e[element_index, int_point_index]
                )
                nodal_sum_of_weighted_s[node_indices] += (
                    nodal_weights[:, None] * int_point_s[element_index, int_point_index]
                )

        # Compute averages:
        mask = nodal_sum_of_weights != 0  # indices of nonzero weights
        nodal_e = numpy.zeros((self.mesh.n_nodes, 6))
        nodal_s = numpy.zeros((self.mesh.n_nodes, 6))
        nodal_e[mask] = nodal_sum_of_weighted_e[mask] / nodal_sum_of_weights[mask, None]
        nodal_s[mask] = nodal_sum_of_weighted_s[mask] / nodal_sum_of_weights[mask, None]

        logger.info(
            f"Nodal strains and stresses computed by L2 projection in {round(clock() - start, 3)} seconds."
        )

        return int_point_e, int_point_s, nodal_e, nodal_s

    @cached_property
    def _precompute_strain_and_stress_by_extrapolation(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Compute strain and stress values at quadrature points and
        extrapolate them to nodes by shape-function extrapolation and
        nodal averaging. This is what Ansys Mechanical does.
        """
        start = clock()

        n_contributions = numpy.zeros(self.mesh.n_nodes, dtype=int)
        int_point_e = numpy.empty((self.mesh.n_elements, self.mesh.n_int_points, 6))
        int_point_s = numpy.empty((self.mesh.n_elements, self.mesh.n_int_points, 6))
        nodal_e = numpy.zeros((self.mesh.n_nodes, 6))
        nodal_s = numpy.zeros((self.mesh.n_nodes, 6))

        match self.mesh.type:
            case ElementType.HEX8 | ElementType.HEX20:
                element_type = ElementType.HEX8
                local_corner_node_indices = numpy.arange(8)
            case ElementType.TET10 | ElementType.TET4:
                element_type = ElementType.TET4
                local_corner_node_indices = numpy.arange(4)
            case _:
                raise NotImplementedError(f"Element type {self.mesh.type} not supported.")

        N = numpy.array(
            [
                utils.N(element_type, point)[local_corner_node_indices]
                for point in utils.INT_POINTS[self.mesh.type]
            ]
        )
        W = numpy.diag(utils.INT_WEIGHTS[self.mesh.type])
        extrapolation_matrix = numpy.linalg.inv(N.T @ W @ N) @ N.T @ W

        for element_index, node_indices in enumerate(self.mesh.connectivity):
            corner_node_indices = node_indices[local_corner_node_indices]
            dofs = utils.dof_indices(node_indices)

            for int_point_index, dN_dxyz in enumerate(self.mesh.dN_dxyz[element_index]):
                int_point_e[element_index, int_point_index] = utils.B(dN_dxyz) @ self.u[dofs]
                int_point_s[element_index, int_point_index] = (
                    self.C(element_index, int_point_index)
                    @ int_point_e[element_index, int_point_index]
                )

            n_contributions[corner_node_indices] += 1
            nodal_e[corner_node_indices] += extrapolation_matrix @ int_point_e[element_index]
            nodal_s[corner_node_indices] += extrapolation_matrix @ int_point_s[element_index]

        mask = n_contributions != 0
        nodal_e[mask] /= n_contributions[mask, None]
        nodal_s[mask] /= n_contributions[mask, None]

        logger.info(
            f"Nodal strains and stresses computed by extrapolation in {round(clock() - start, 3)} seconds."
        )

        return int_point_e, int_point_s, nodal_e, nodal_s

    def get_strain_at_int_points(self, projection_method: ProjectionMethod) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Strain values at integration points as an E x I x 6 array
            where E is the number of elements and I is the number of
            integration points per element.
        """
        match projection_method:
            case ProjectionMethod.L2:
                return self._precompute_strain_and_stress_by_L2_projection[0]
            case ProjectionMethod.ANSYS:
                return self._precompute_strain_and_stress_by_extrapolation[0]
            case _:
                raise ValueError(f"Unknown projection method: {projection_method}")

    def get_stress_at_int_points(
        self,
        projection_method: ProjectionMethod = ProjectionMethod.L2,
    ) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Stress values at integration points as an E x I x 6 array
            where E is the number of elements and I is the number of
            integration points per element.
        """
        match projection_method:
            case ProjectionMethod.L2:
                return self._precompute_strain_and_stress_by_L2_projection[1]
            case ProjectionMethod.ANSYS:
                return self._precompute_strain_and_stress_by_extrapolation[1]
            case _:
                raise ValueError(f"Unknown projection method: {projection_method}")

    def get_strain_at_nodes(
        self,
        projection_method: ProjectionMethod = ProjectionMethod.L2,
    ) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Nodal strain values as an N x 6 array where N is the number
            of nodes.
        """
        match projection_method:
            case ProjectionMethod.L2:
                return self._precompute_strain_and_stress_by_L2_projection[2]
            case ProjectionMethod.ANSYS:
                return self._precompute_strain_and_stress_by_extrapolation[2]
            case _:
                raise ValueError(f"Unknown projection method: {projection_method}")

    def get_stress_at_nodes(
        self,
        projection_method: ProjectionMethod = ProjectionMethod.L2,
    ) -> NDArray:
        """
        Returns
        -------
        numpy.ndarray
            Nodal stress values as an N x 6 array where N is the number
            of nodes.
        """
        match projection_method:
            case ProjectionMethod.L2:
                return self._precompute_strain_and_stress_by_L2_projection[3]
            case ProjectionMethod.ANSYS:
                return self._precompute_strain_and_stress_by_extrapolation[3]
            case _:
                raise ValueError(f"Unknown projection method: {projection_method}")

    def get_u_at_point(self, point: NDArray, k: int = 27) -> NDArray:
        """Interpolate the displacement from the global displacement vector.

        Parameters
        ----------
        point : numpy.ndarray
            Point where the stress shall be computed.
        k : int
            Number of candidate elements to query from the k-d tree.

        Returns
        -------
        numpy.ndarray
            Displacement vector at `point`.
        """
        element_index, rst = self.mesh.get_rst(point, k)
        dof_indices = utils.dof_indices(node_indices=self.mesh.connectivity[element_index])
        return utils.N(self.mesh.type, rst) @ self.u[dof_indices].reshape(-1, 3)

    def get_stress_at_point(self, point: NDArray, k: int = 27) -> NDArray:
        """Interpolate the local stress from stress values at the nodes.

        Parameters
        ----------
        point : numpy.ndarray
            Point where the stress shall be computed.
        k : int
            Number of candidate elements to query from the k-d tree.

        Returns
        -------
        numpy.ndarray
            Stress vector at `point`.

        Notes
        -----
        Stresses at nodes are extrapolated from stresses at Gauss points.
        Directly extrapolating from Gauss points may be more accurate
        but less smooth than interpolating from nodes using the
        element's shape function. As elements are assumed to be small
        and fiber paths shall be as smooth as possible, stresses are
        interpolated from nodes for this application.
        """
        element_index, rst = self.mesh.get_rst(point, k)
        stresses = self.get_stress_at_nodes()[self.mesh.connectivity[element_index]]
        shape_function_values = utils.N(self.mesh.type, rst)
        return numpy.average(stresses, axis=0, weights=shape_function_values)
