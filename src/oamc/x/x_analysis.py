from time import perf_counter as timer
import logging

import numpy
import scipy
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint, minimize

from oamc.fea.analysis import Analysis
from oamc.fea.bc import BC
from oamc.fea.mesh import Mesh
from oamc.fea import utils as fea_utils
from oamc.x.x_material import XMaterial
from oamc.x import utils as x_utils

logger = logging.getLogger(__name__)


class XAnalysis(Analysis[XMaterial]):
    """..."""

    def __init__(
        self,
        mesh: Mesh,
        material: XMaterial,
        dbc: list[BC],
        nbc: list[BC],
    ):
        super().__init__(mesh=mesh, material=material, dbc=dbc, nbc=nbc)

        # Design variables in the format [psi 1 at node 1, psi 1 at node 2, ..., psi 2 at node 1, psi 2 at node 2, ...]:
        self.psi = numpy.zeros(2 * self.mesh.ndof, dtype=numpy.float64)

        # Constant spacing of the level surfaces:
        self.DELTA_1 = 1
        self.DELTA_2 = 1
        # The values don't matter, they just have to be consistent.

    def K(self) -> scipy.sparse.csc_array:
        """Assemble the global stiffness matrix."""

        start = timer()

        rows, columns, values = [], [], []

        for element, nodes in enumerate(self.mesh.element_connectivity):
            # Determine degrees of freedom (DOF) per element:
            DOF = numpy.repeat(nodes, 3) * 3 + numpy.tile([0, 1, 2], nodes.size)

            # Initialize element stiffness matrix:
            Ke = numpy.zeros(shape=(DOF.size, DOF.size), dtype=numpy.float64)

            # Compute element stiffness matrix:
            for jac_N, w_det_J in zip(
                self.mesh.dN_dxyz[element],
                self.mesh.w_det_dxyz_drst[element],
            ):
                # Compute strain-displacement matrix:
                B = fea_utils.B(jac_N)

                # Compute material stiffness matrix:
                a, b = jac_N.T @ self.psi[0, nodes], jac_N.T @ self.psi[1, nodes]
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
            rows.append(numpy.repeat(DOF, DOF.size))
            columns.append(numpy.tile(DOF, DOF.size))
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

    def compliance(self) -> float:
        return self.f @ self.u

    def grad_compliance(self) -> numpy.ndarray:
        grad = numpy.zeros_like(self.psi)

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
                a = jac_N.T @ self.psi[nodes]  # = d psi_1 / d xyz
                b = jac_N.T @ self.psi[nodes + self.mesh.ndof]  # = d psi_2 / d xyz
                c = numpy.cross(a, b)
                nc = numpy.linalg.norm(c)

                if nc == 0:
                    logger.warning("Zero gradient.")
                    continue

                # Fiber tangent direction:
                t = c / nc

                # Fiber volume fraction:
                v = self.material.fiber_diameter * nc / self.DELTA1 / self.DELTA2

                dc_da = -x_utils.skew(b)
                dc_db = +x_utils.skew(a)

                dnc_dc = t  # = c / nc

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

                for i, node in enumerate(nodes):
                    grad[node] -= (
                        strain.T
                        @ self.material.dC(v=v, t=t, dv=dv_dpsi_1[i], dt=dt_dpsi_1[i])
                        @ strain
                        * w_det_J
                    )
                    grad[node + self.mesh.ndof] -= (
                        strain.T
                        @ self.material.dC(v=v, t=t, dv=dv_dpsi_2[i], dt=dt_dpsi_2[i])
                        @ strain
                        * w_det_J
                    )

        return grad

    def min_spacing(self) -> float:
        # TODO: Implement differentiable approximation of minimum fiber spacing using KS aggregation.
        ...

    def grad_min_spacing(self) -> NDArray:
        # TODO: Implement gradient of minimum fiber spacing.
        ...

    def total_length(self) -> float:
        # TODO: Implement computation of total fiber length.
        ...

    def grad_total_length(self) -> NDArray:
        # TODO: Implement gradient of total fiber length.
        ...

    def psi_min(self) -> float:
        # TODO: Implement differentiable approximation of the minimum value of psi using KS aggregation.
        ...

    def grad_psi_min(self) -> NDArray:
        # TODO: Implement gradient of the minimum value of psi
        ...

    def optimize(
        self,
        min_spacing: float,
        max_fiber_length: float | None = None,
        max_fiber_volume_fraction: float | None = None,
        max_weight: float | None = None,
    ) -> NDArray:
        """
        :param min_spacing: minimum allowed fiber spacing
        :param max_fiber_length: maximum allowed total fiber length
        :param max_fiber_volume_fraction: maximum allowed fiber volume fraction
        :param max_weight: maximum allowed weight of the part
        :return: optimized design variables
        """

        if min_spacing <= 0:
            raise ValueError("Minimum spacing must be positive.")
        if max_fiber_length is not None and max_fiber_length <= 0:
            raise ValueError("Maximum fiber length must be positive.")
        if max_fiber_volume_fraction is not None:
            raise NotImplementedError(
                "Maximum fiber volume fraction constraint is not yet implemented."
            )
        if max_weight is not None:
            raise NotImplementedError("Maximum weight constraint is not yet implemented.")

        # TODO: Initialize self.psi based on max. principal stress and max. fiber volume fraction.

        result = minimize(
            fun=self.compliance,
            jac=self.grad_compliance,
            x0=self.psi,
            method="L-BFGS-B",
            constraints=[
                NonlinearConstraint(
                    fun=self.min_spacing,
                    lb=min_spacing,
                    ub=numpy.inf,
                    jac=self.grad_min_spacing,
                ),
                NonlinearConstraint(
                    fun=self.total_length,
                    lb=-numpy.inf,
                    ub=max_fiber_length,
                    jac=self.grad_total_length,
                ),
                NonlinearConstraint(
                    fun=self.psi_min,
                    lb=0,
                    ub=numpy.inf,
                    jac=self.grad_psi_min,
                ),
            ],
        )
        return result.x
