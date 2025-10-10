import logging
from functools import cached_property
from time import perf_counter as timer
from typing import Generic, TypeVar

import numpy
import pypardiso
import scipy.sparse
import scipy.sparse.linalg

# from numba import float64, njit
from numpy.typing import NDArray

from oamc.fem import utils
from oamc.fem.bc import BC
from oamc.fem.material import Material
from oamc.fem.mesh import Mesh

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
#     return numpy.zeros((6, 6), dtype=numpy.float64)


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
#     return numpy.zeros((6, 6), dtype=numpy.float64)


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
#     return numpy.zeros((6, 6), dtype=numpy.float64)


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
#     return numpy.zeros((6, 6), dtype=numpy.float64)

# As Material has various subclasses:
MATERIAL = TypeVar("MATERIAL", bound=Material)


class Analysis(Generic[MATERIAL]):
    """A linear static finite-element analysis."""

    def __init__(
        self,
        mesh: Mesh,
        material: MATERIAL,
        dbc: list[BC],
        nbc: list[BC],
    ):
        self.mesh = mesh
        self.material = material
        self.dbc = dbc
        self.nbc = nbc

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "C" not in cls.__dict__:
            logger.warning(
                "Computation of local material stiffness matrix not overridden by subclass."
            )

    def __setattr__(self, name, value):
        self.__dict__.pop("f", None)
        self.__dict__.pop("u", None)
        self.__dict__.pop("_precompute_e_and_s", None)
        super().__setattr__(name, value)

    @cached_property
    def f(self) -> NDArray:
        """Return the global force vector."""

        f = numpy.zeros(self.mesh.nodes.shape[0] * 3)

        # Apply Neumann boundary conditions:
        for nbc in self.nbc:
            f[nbc.node * 3 + nbc.direction.value] += nbc.value

        return f

    @cached_property
    def u(self) -> NDArray:
        """Compute the nodal displacement vector."""

        start = timer()

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

        logger.info(f"Boundary conditions applied in {round(timer() - start, 3)} seconds.")

        start = timer()

        u_f = pypardiso.spsolve(K_ff, f_f)

        # Alternative solvers:

        # ilu = scipy.sparse.linalg.spilu(k_ff)
        # preconditioner = scipy.sparse.linalg.LinearOperator(k_ff.shape, ilu.solve)
        # u_f, info = scipy.sparse.linalg.cg(
        #     A=k_ff,
        #     b=f_f,
        #     atol=1e-10,
        #     maxiter=2000,
        #     M=preconditioner,
        # )
        # logger.info(f"SciPy CG solver exit code: {info}")

        # u_f = scipy.sparse.linalg.spsolve(k_ff, f_f)

        u = numpy.zeros(n)
        u[free] = u_f
        u[constrained] = u_dbc[constrained]

        logger.info(f"System solved in {round(timer() - start, 3)} seconds.")

        return u.reshape

    @cached_property
    def _precompute_e_and_s(self) -> tuple[NDArray, NDArray]:
        start = timer()

        # Accumulators for nodal projection:
        nodal_sum_of_weights = numpy.zeros(self.mesh.node_count)
        nodal_sum_of_weighted_e = numpy.zeros((self.mesh.node_count, 6))
        nodal_sum_of_weighted_s = numpy.zeros((self.mesh.node_count, 6))

        N_at_int_points = tuple(
            [
                utils.N(self.mesh.element_type, point)
                for point in utils.INT_POINTS[self.mesh.element_type]
            ]
        )

        for element_index, node_indices in enumerate(self.mesh.element_connectivity):
            dofs = numpy.repeat(node_indices, 3) * 3 + numpy.tile([0, 1, 2], len(node_indices))
            u_element = self.u[dofs]

            for int_point_index, (N_values, dN_dxyz, w_det_J) in enumerate(
                zip(
                    N_at_int_points,
                    self.mesh.dN_dxyz[element_index],
                    self.mesh.w_det_dxyz_drst[element_index],
                )
            ):
                e = utils.B(dN_dxyz) @ u_element
                s = self.C(element_index, int_point_index) @ e

                nodal_weights = N_values * w_det_J
                nodal_sum_of_weights[node_indices] += nodal_weights
                nodal_sum_of_weighted_e[node_indices] += nodal_weights[:, None] * e
                nodal_sum_of_weighted_s[node_indices] += nodal_weights[:, None] * s

        # Compute averages:
        mask = nodal_sum_of_weights > 0  # indices of nonzero weights
        nodal_strain = numpy.zeros((self.mesh.node_count, 6))
        nodal_stress = numpy.zeros((self.mesh.node_count, 6))
        nodal_strain[mask] = nodal_sum_of_weighted_e[mask] / nodal_sum_of_weights[mask, None]
        nodal_stress[mask] = nodal_sum_of_weighted_s[mask] / nodal_sum_of_weights[mask, None]

        logger.info(f"Nodal strains and stresses computed in {round(timer() - start, 3)} seconds.")

        return nodal_strain, nodal_stress

    @property
    def e(self) -> NDArray:
        """Compute strain values at nodes based on the displacement vector.

        :return: strain values at nodes as an N x 6 array, where N is the
            number of nodes
        """
        return self._precompute_e_and_s[0]

    @property
    def s(self) -> NDArray:
        """Compute stress values at nodes based on the displacement vector.

        :return: stress values at nodes as an N x 6 array, where N is the
            number of nodes
        """
        return self._precompute_e_and_s[1]

    def C(
        self,
        element_index: int,
        int_point_index: int,
    ) -> NDArray:
        """Return the material stiffness matrix."""
        return self.material.C

    def K(self) -> scipy.sparse.csc_array:
        """Assemble the global stiffness matrix."""

        start = timer()

        rows, columns, values = [], [], []

        for element_index, node_indices in enumerate(self.mesh.element_connectivity):
            # Determine degrees of freedom (DOF) per element:
            dofs = numpy.repeat(node_indices, 3) * 3 + numpy.tile([0, 1, 2], node_indices.size)

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

        # Assemble the global stiffness matrix in COO format, then convert to CSC format:
        K = scipy.sparse.coo_array(
            arg1=(
                numpy.concatenate(values),
                (
                    numpy.concatenate(rows),
                    numpy.concatenate(columns),
                ),
            ),
            shape=(self.mesh.dof_count, self.mesh.dof_count),
        ).tocsc()

        K.sort_indices()

        logger.info(f"Global stiffness matrix assembled in {round(timer() - start, 3)} seconds.")

        return K

    def stress_at(self, point: NDArray, k: int = 27) -> NDArray:
        """Interpolate the stress at `point` from stresses at nodes.

        NOTE: Stresses at nodes are extrapolated from stresses at Gauss points.
        Directly extrapolating from Gauss points may be more accurate but less
        smooth than interpolating from nodes using the element's shape function.
        As elements are assumed to be small and fiber paths shall be as smooth
        as possible, stresses are interpolated from nodes for this application.

        :param point: point where the stress shall be computed
        :param k: number of candidate elements to query from the k-d tree
        :return: stress vector at `point`
        """
        element_index, rst = self.mesh.rst(point, k)
        stresses = self.s[self.mesh.element_connectivity[element_index]]
        shape_function_values = utils.N(self.mesh.element_type, rst)
        return numpy.average(stresses, axis=0, weights=shape_function_values)
