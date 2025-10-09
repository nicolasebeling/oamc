from time import perf_counter as timer
from typing import Callable
import logging

import numpy
import scipy
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint, minimize

from oamc.constants import NODE_COUNT_FROM_ELEMENT_TYPE
from oamc.fea.analysis import Analysis
from oamc.fea.bc import BC
from oamc.fea.mesh import Mesh
from oamc.fea import utils as fea_utils
from oamc.x.composite_material import CompositeMaterial
from oamc.x import utils

logger = logging.getLogger(__name__)


class ConstraintWrapper:
    def __init__(self, fun_and_jac: Callable[[NDArray], tuple[float, NDArray]]):
        self.fun_and_jac = fun_and_jac
        self._last_x = None
        self._last_fun = None
        self._last_jac = None

    def fun(self, x: NDArray):
        if self._last_x is None or not numpy.allclose(x, self._last_x):
            self._last_fun, self._last_jac = self.fun_and_jac(x)
            self._last_x = numpy.copy(x)
        return self._last_fun

    def jac(self, x: NDArray):
        if self._last_x is None or not numpy.allclose(x, self._last_x):
            self._last_fun, self._last_jac = self.fun_and_jac(x)
            self._last_x = numpy.copy(x)
        return self._last_jac


class CompositeAnalysis(Analysis[CompositeMaterial]):
    """..."""

    def __init__(
        self,
        mesh: Mesh,
        material: CompositeMaterial,
        dbc: list[BC],
        nbc: list[BC],
    ):
        super().__init__(mesh=mesh, material=material, dbc=dbc, nbc=nbc)

        # Design variables in the format [p_1 at node 1, p_1 at node 2, ..., p_2 at node 1, p_2 at node 2, ...]:
        self.p = numpy.zeros(2 * self.mesh.n_dof, dtype=numpy.float64)

        # Constant spacing of the level surfaces:
        self.D1 = 1
        self.D2 = 1
        # The values don't matter, they just have to be consistent.

    def K(self) -> scipy.sparse.csc_array:
        """Assemble the global stiffness matrix."""

        start = timer()

        rows, columns, values = [], [], []

        for element, nodes in enumerate(self.mesh.element_connectivity):
            # Determine degrees of freedom (DOF) per element:
            dof = numpy.repeat(nodes, 3) * 3 + numpy.tile([0, 1, 2], nodes.size)

            # Initialize element stiffness matrix:
            Ke = numpy.zeros(shape=(dof.size, dof.size), dtype=numpy.float64)

            # Compute element stiffness matrix:
            for jac_N, w_det_J in zip(
                self.mesh.dN_dxyz[element],
                self.mesh.w_det_dxyz_drst[element],
            ):
                # Compute strain-displacement matrix:
                B = fea_utils.B(jac_N)

                # Compute material stiffness matrix:
                a = jac_N.T @ self.p[nodes]
                b = jac_N.T @ self.p[self.mesh.n_dof + nodes]
                c = numpy.cross(a, b)
                norm_c = numpy.linalg.norm(c)
                if norm_c == 0:
                    C = self.material.C_m()
                else:
                    # Compute fiber volume fraction:
                    t = c / norm_c
                    v = self.material.fiber_diameter * norm_c / self.DELTA1 / self.DELTA2
                    C = self.material.C(v=v, t=t)

                # Add to element stiffness matrix:
                Ke += (B.T @ C @ B) * w_det_J

            # Add to global stiffness matrix:
            rows.append(numpy.repeat(dof, dof.size))
            columns.append(numpy.tile(dof, dof.size))
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
            shape=(self.mesh.n_dof, self.mesh.n_dof),
        ).tocsc()

        K.sort_indices()

        logger.info(f"Global stiffness matrix assembled in {round(timer() - start, 3)} seconds.")

        return K

    def __setattr__(self, name, value):
        if name == "p" and numpy.array_equal(self.__dict__.get(name, None), value):
            return
        super().__setattr__(name, value)

    def compliance(self, p: NDArray) -> float:
        self.p = p

        return self.f @ self.u

    def grad_compliance(self, p: NDArray) -> numpy.ndarray:
        self.p = p

        grad = numpy.zeros_like(self.p)

        for element, nodes in enumerate(self.mesh.element_connectivity):
            # Determine degrees of freedom (DOF) per element:
            dof = numpy.repeat(nodes, 3) * 3 + numpy.tile([0, 1, 2], nodes.size)

            for jac_N, w_det_J in zip(
                self.mesh.dN_dxyz[element],
                self.mesh.w_det_dxyz_drst[element],
            ):
                # Compute strain:
                strain = fea_utils.B(jac_N) @ self.u[dof]

                # Compute material stiffness matrix derivatives:
                a = jac_N.T @ self.p[nodes]  # = d psi_1 / d xyz
                b = jac_N.T @ self.p[nodes + self.mesh.n_dof]  # = d psi_2 / d xyz
                c = numpy.cross(a, b)
                nc = numpy.linalg.norm(c)

                if nc == 0:
                    logger.warning("Zero gradient.")
                    continue

                # Fiber tangent direction:
                t = c / nc

                # Fiber volume fraction:
                v = self.material.fiber_diameter * nc / self.DELTA1 / self.DELTA2

                dc_da = utils.skew(-b)
                dc_db = utils.skew(a)

                dnc_dc = t.T  # = c / nc (the transpose doesn't matter as it's a 1D array)

                dnc_da = dnc_dc @ dc_da
                dnc_db = dnc_dc @ dc_db

                dt_dc = (numpy.eye(3) - numpy.outer(t, t)) / nc

                dt_da = dt_dc @ dc_da
                dt_db = dt_dc @ dc_db

                fiber_area = numpy.pi * (self.material.fiber_diameter / 2) ** 2

                # d a / d psi_1 = d b / d psi_2 = jac_N.T
                # fiber_area / self.DELTA1 / self.DELTA2 = d v / d nc
                dv_dpsi_1 = fiber_area / self.DELTA1 / self.DELTA2 * dnc_da @ jac_N.T
                dv_dpsi_2 = fiber_area / self.DELTA1 / self.DELTA2 * dnc_db @ jac_N.T

                dt_dpsi_1 = dt_da @ jac_N.T
                dt_dpsi_2 = dt_db @ jac_N.T

                for node_local, node_global in enumerate(nodes):
                    grad[node_global] -= (
                        strain.T
                        @ self.material.dC(
                            v=v,
                            t=t,
                            dv=dv_dpsi_1[node_local],
                            dt=dt_dpsi_1[node_local],
                        )
                        @ strain
                        * w_det_J
                    )
                    grad[node_global + self.mesh.n_dof] -= (
                        strain.T
                        @ self.material.dC(
                            v=v,
                            t=t,
                            dv=dv_dpsi_2[node_local],
                            dt=dt_dpsi_2[node_local],
                        )
                        @ strain
                        * w_det_J
                    )

        return grad

    def min_spacing(
        self,
        p: NDArray,
        inner_r: float = -30,
        outer_r: float = -30,
    ) -> tuple[float, NDArray]:
        """
        Return the approximate minimum fiber spacing and its gradient.

        :param p: design variables (scalar fields at nodes)
        :param inner_r: inner KS aggregation parameter (for min.
            spacing candidates at each integration point)
        :param outer_r: outer KS aggregation parameter (for KS
            aggregations of min. spacing candidates at each integration point)
        :return: approximate minimum fiber spacing and its gradient
        """
        if inner_r >= 0 or outer_r >= 0:
            raise ValueError("KS aggregation parameters must be negative for minimum aggregation.")

        self.p = p

        outer_constraints = numpy.empty(
            shape=(self.mesh.n_elements * fea_utils.N_INT_POINTS[self.mesh.element_type],),
            dtype=numpy.float64,
        )

        # Linear combinations of lattice basis vectors to consider for minimum spacing evaluation:
        linear_combinations = numpy.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [1, -1],
            ]
        )

        # Constraint evaluation loop:
        for element_index, node_indices in enumerate(self.mesh.element_connectivity):
            for int_point_index, jac_N in enumerate(self.mesh.dN_dxyz[element_index]):
                a = jac_N.T @ self.p[node_indices]  # = d psi_1 / d xyz
                b = jac_N.T @ self.p[node_indices + self.mesh.n_dof]  # = d psi_2 / d xyz
                c = numpy.cross(a, b)
                norm_c = numpy.linalg.norm(c)
                if norm_c == 0:
                    logger.warning("Zero gradient.")
                    continue
                t = c / norm_c  # fiber tangent direction
                d = numpy.vstack(
                    (
                        self.D1 * numpy.cross(t, b) / norm_c,
                        self.D2 * numpy.cross(t, a) / norm_c,
                    )
                )  # rows = vectors from one fiber to the next in the two level surfaces

                inner_constraints = numpy.linalg.norm(linear_combinations @ d, axis=1)

                outer_constraints[element_index + int_point_index] = utils.ks(
                    f=inner_constraints,
                    r=inner_r,
                )

        outer_exponents = outer_r * outer_constraints
        outer_max_exponent = numpy.max(outer_exponents)
        outer_exponentials = numpy.exp(outer_exponents - outer_max_exponent)
        outer_grad_weights = outer_exponentials / numpy.sum(outer_exponentials)
        outer_ks = (outer_max_exponent + numpy.log(numpy.sum(outer_exponentials))) / outer_r

        outer_grad_ks = numpy.empty_like(self.p)

        elem_grad_ks = numpy.empty(
            shape=(2, NODE_COUNT_FROM_ELEMENT_TYPE[self.mesh.element_type]),
            dtype=numpy.float64,
        )

        # Gradient evaluation loop:
        for element_index, node_indices in enumerate(self.mesh.element_connectivity):
            elem_grad_ks.fill(0)
            for int_point_index, jac_N in enumerate(self.mesh.dN_dxyz[element_index]):
                a = jac_N.T @ self.p[node_indices]  # = d psi_1 / d xyz
                b = jac_N.T @ self.p[node_indices + self.mesh.n_dof]  # = d psi_2 / d xyz
                c = numpy.cross(a, b)
                norm_c = numpy.linalg.norm(c)
                if norm_c == 0:
                    logger.warning("Zero gradient.")
                    continue
                t = c / norm_c  # fiber tangent direction
                d = numpy.vstack(
                    (
                        self.D1 * numpy.cross(t, b) / norm_c,
                        self.D2 * numpy.cross(t, a) / norm_c,
                    )
                )  # rows = vectors from one fiber to the next in the two level surfaces

                inner_constraints = numpy.linalg.norm(linear_combinations @ d, axis=1)

                par_d_par_grad_p_1 = numpy.empty((linear_combinations.shape[0], 3))
                par_d_par_grad_p_2 = numpy.empty((linear_combinations.shape[0], 3))

                for i in range(linear_combinations.shape[0]):
                    m = linear_combinations[i, 0]
                    n = linear_combinations[i, 1]

                    u = self.D1 * m * b + self.D2 * n * a

                    par_d_par_grad_p_1[i] = (
                        (self.D2 * n * u / d[i])
                        - d[i] * (numpy.linalg.norm(b) ** 2 * a - (a @ b) * b)
                    ) / (norm_c**2)
                    par_d_par_grad_p_2[i] = (
                        (self.D1 * m * u / d[i])
                        - d[i] * (numpy.linalg.norm(a) ** 2 * b - (a @ b) * a)
                    ) / (norm_c**2)

                inner_exponents = inner_r * inner_constraints
                inner_max_exponent = numpy.max(inner_exponents)
                inner_exponentials = numpy.exp(inner_exponents - inner_max_exponent)

                inner_grad_weights = inner_exponentials / numpy.sum(inner_exponentials)
                par_ks_par_p_1 = jac_N @ (par_d_par_grad_p_1.T @ inner_grad_weights)
                par_ks_par_p_2 = jac_N @ (par_d_par_grad_p_2.T @ inner_grad_weights)

                elem_grad_ks[0] += (
                    outer_grad_weights[element_index + int_point_index] * par_ks_par_p_1
                )
                elem_grad_ks[1] += (
                    outer_grad_weights[element_index + int_point_index] * par_ks_par_p_2
                )

            # Add elemental to global (outer) gradient:
            outer_grad_ks[node_indices] += elem_grad_ks[0]
            outer_grad_ks[node_indices + self.mesh.n_dof] += elem_grad_ks[1]

        return outer_ks, outer_grad_ks

    def total_length(self, p: NDArray) -> tuple[float, NDArray]:
        self.p = p

        grad = numpy.zeros_like(self.p)

        int_norm_c = 0
        for element, nodes in enumerate(self.mesh.element_connectivity):
            # Determine degrees of freedom (DOF) per element:
            dof = numpy.repeat(nodes, 3) * 3 + numpy.tile([0, 1, 2], nodes.size)

            for jac_N, w_det_J in zip(
                self.mesh.dN_dxyz[element],
                self.mesh.w_det_dxyz_drst[element],
            ):
                a = jac_N.T @ self.p[nodes]  # = d psi_1 / d xyz
                b = jac_N.T @ self.p[nodes + self.mesh.n_dof]  # = d psi_2 / d xyz
                c = numpy.cross(a, b)
                norm_c = numpy.linalg.norm(c)
                if norm_c == 0:
                    logger.warning("Zero gradient.")
                    continue

                int_norm_c += norm_c * w_det_J

                dc_da = utils.skew(-b)
                dc_db = utils.skew(a)

                dnc_dc = (c / norm_c).T  # (the transpose doesn't matter as it's a 1D array)

                dnc_da = dnc_dc @ dc_da
                dnc_db = dnc_dc @ dc_db

                # d a / d psi_1 = d b / d psi_2 = jac_N.T
                # 1 / self.DELTA1 / self.DELTA2 = d l / d nc
                dl_dpsi_1 = jac_N.T @ dnc_da / self.DELTA1 / self.DELTA2
                dl_dpsi_2 = jac_N.T @ dnc_db / self.DELTA1 / self.DELTA2

                grad[dof] += dl_dpsi_1 * w_det_J
                grad[dof + self.mesh.n_dof] += dl_dpsi_2 * w_det_J

        return int_norm_c / self.D1 / self.D2, grad

    def min_psi(self, p: NDArray) -> tuple[float, NDArray]:
        return utils.ks(p, -30)

    def optimize(
        self,
        min_spacing: float,
        max_fiber_length: float | None = None,
        max_fiber_volume_fraction: float | None = None,
        max_fiber_weight: float | None = None,
    ) -> NDArray:
        """
        :param min_spacing: minimum allowed fiber spacing
        :param max_fiber_length: maximum allowed total fiber length
        :param max_fiber_volume_fraction: maximum allowed fiber volume fraction
        :param max_fiber_weight: maximum allowed fiber weight
        :return: optimized design variables
        """

        if min_spacing <= 0:
            raise ValueError("Minimum fiber spacing must be positive.")
        if max_fiber_length is not None and max_fiber_length <= 0:
            raise ValueError("Maximum fiber length must be positive.")
        if max_fiber_volume_fraction is not None:
            raise NotImplementedError("Fiber volume fraction constraint is not yet implemented.")
        if max_fiber_weight is not None:
            raise NotImplementedError("Fiber weight constraint is not yet implemented.")

        # TODO: Initialize self.p based on max. principal stress and max. fiber volume fraction.

        min_spacing_wrapper = ConstraintWrapper(self.min_spacing)
        total_length_wrapper = ConstraintWrapper(self.total_length)
        min_psi_wrapper = ConstraintWrapper(self.min_psi)

        result = minimize(
            fun=self.compliance,
            jac=self.grad_compliance,
            x0=self.p,
            method="L-BFGS-B",
            constraints=[
                NonlinearConstraint(
                    fun=min_spacing_wrapper.fun,
                    lb=min_spacing,
                    ub=numpy.inf,
                    jac=min_spacing_wrapper.jac,
                ),
                NonlinearConstraint(
                    fun=total_length_wrapper.fun,
                    lb=-numpy.inf,
                    ub=max_fiber_length,
                    jac=total_length_wrapper.jac,
                ),
                NonlinearConstraint(
                    fun=min_psi_wrapper.fun,
                    lb=0,
                    ub=numpy.inf,
                    jac=min_psi_wrapper.jac,
                ),
            ],
        )
        return result.x

    def integrate_fiber_paths(self) -> NDArray:
        # TODO: Integrate fiber paths.
        ...
