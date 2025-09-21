import logging
from functools import cached_property
from time import perf_counter as timer
from typing import TypeVar, Generic

import numpy
import pypardiso
import scipy.sparse
import scipy.sparse.linalg

# from numba import float64, njit
from numpy.typing import NDArray

from oamc.constants import NODE_COUNT_FROM_ELEMENT_TYPE
from oamc.fea import utils
from oamc.fea.bc import BC
from oamc.fea.material import Material
from oamc.fea.mesh import Mesh

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

        u_f = pypardiso.spsolve(K_ff, f_f)

        u = numpy.zeros(n)
        u[free] = u_f
        u[constrained] = u_dbc[constrained]

        logger.info(f"System solved in {round(timer() - start, 3)} seconds.")

        return u.reshape

    @cached_property
    def e(self) -> NDArray:
        """Compute strain values at nodes based on the displacement vector.

        :return: strain values at nodes as an N x 6 array, where N is the
            number of nodes
        """
        raise NotImplementedError("Strain computation is not yet implemented.")

    @cached_property
    def s(self) -> NDArray:
        """Compute stress values at nodes based on the displacement vector.

        :return: stress values at nodes as an N x 6 array, where N is the
            number of nodes
        """
        raise NotImplementedError("Stress computation is not yet implemented.")

    def __setattr__(self, name, value):
        self.__dict__.pop("f", None)
        self.__dict__.pop("u", None)
        self.__dict__.pop("e", None)
        self.__dict__.pop("s", None)
        self.__dict__[name] = value

    def K(self) -> scipy.sparse.csc_array:
        """Assemble the global stiffness matrix."""

        start = timer()

        rows, columns, values = [], [], []

        for element, nodes in enumerate(self.mesh.element_connectivity):
            # Numer of degrees of freedom per element:
            ndof = NODE_COUNT_FROM_ELEMENT_TYPE[self.mesh.element_type] * 3
            Ke = numpy.zeros(shape=(ndof, ndof), dtype=numpy.float64)

            # Compute element stiffness matrix:
            for jac_N, w_det_J in zip(
                self.mesh.dN_dxyz[element], self.mesh.w_det_dxyz_drst[element]
            ):
                B = utils.B(jac_N)
                Ke += (B.T @ self.material.C @ B) * w_det_J

            # Add to global stiffness matrix:
            dofs = numpy.repeat(nodes, 3) * 3 + numpy.tile([0, 1, 2], nodes.size)
            rows.append(numpy.repeat(dofs, dofs.size))
            columns.append(numpy.tile(dofs, dofs.size))
            values.append(Ke.ravel())

        # Assemble the global stiffness matrix in COO format, then convert to CSC format:
        K = scipy.sparse.coo_array(
            arg1=(
                numpy.concatenate(values),
                (
                    numpy.concatenate(rows),
                    numpy.concatenate(columns),
                ),
            ),
            shape=(self.mesh.ndof, self.mesh.ndof),
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
        shape_functions = utils.N(self.mesh.element_type, rst)
        return numpy.average(stresses, axis=0, weights=shape_functions)
