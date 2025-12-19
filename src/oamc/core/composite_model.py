"""
Classes
-------
CompositeModel
"""

import logging
from collections import defaultdict
from itertools import chain
from os import makedirs
from pathlib import Path
from time import perf_counter as clock
from typing import Callable, Literal

import numpy
import pypardiso
import scipy.optimize
import scipy.sparse
from numpy.typing import NDArray

from oamc.core.composite_material import CompositeMaterial
from oamc.enums import AngleUnit, Direction
from oamc.fem import utils as utils
from oamc.fem.bc import BC
from oamc.fem.mesh import SolidMesh, SurfaceMesh
from oamc.fem.model import SolidModel
from oamc.fiber import Fiber
from oamc.utils.math import skew
from oamc.utils.mechanics import principal_stress, vector_to_tensor
from oamc.utils.optimization import FunctionCache, ks
from oamc.utils.vtk import compute_int_isosurface_intersections

logger = logging.getLogger(__name__)


class CompositeModel(SolidModel[CompositeMaterial]):
    def __init__(
        self,
        mesh: SolidMesh,
        mold: SurfaceMesh,
        material: CompositeMaterial,
        dbc: list[BC],
        nbc: list[BC],
        fiber_diameter: float,
        layer_height: float,
    ):
        """
        Create a new CompositeModel instance.

        Parameters
        ----------
        mesh : oamc.fem.Mesh
            Finite-element mesh.
        material : oamc.x.CompositeMaterial
            Composite material.
        dbc : list of oamc.fem.BC
            Dirichlet (essential) boundary consitions.
        nbc : list of oamc.fem.BC
            Neumann (natural) boundary conditions.
        fiber_diameter : float
            Diameter of the fiber filament before compaction.
        layer_height : float
            Height of the FDM printed layers.
        """
        super().__init__(
            mesh=mesh,
            material=material,
            dbc=dbc,
            nbc=nbc,
        )

        self.mold = mold

        if fiber_diameter <= 0:
            raise ValueError("Fiber diameter must be positive.")
        self.fiber_diameter = fiber_diameter
        self.fiber_area = numpy.pi * (fiber_diameter / 2) ** 2

        self.layer_height = layer_height

        # Uninitialized design variables:
        self.p = numpy.zeros(shape=self.mesh.n_nodes, dtype=float)
        self.q = numpy.zeros(shape=self.mesh.n_nodes, dtype=float)

        self.fibers: dict[int, list[Fiber]] | None = None

    _DEP_MAP = SolidModel._merge_dependencies(
        {
            "p": [
                "K",
                "u",
                "_precompute_strain_and_stress_by_L2_projection",
                "_precompute_strain_and_stress_by_extrapolation",
            ],
            "q": [
                "K",
                "u",
                "_precompute_strain_and_stress_by_L2_projection",
                "_precompute_strain_and_stress_by_extrapolation",
            ],
        }
    )

    def update_p(self, p: NDArray) -> None:
        if not numpy.allclose(self.p, p):
            self.p = p

    def get_distance_to_mold(self, points: NDArray) -> float:
        return self.mold.get_distances(points)

    def C(
        self,
        element_index: int,
        int_point_index: int,
    ) -> NDArray:
        """
        Compute the material stiffness matrix.

        Parameters
        ----------
        element_index : int
            Element where the stiffness matrix shall be computed.
        int_point_index : int
            Integration point where the stiffness matrix shall be
            computed.
        """

        jac_N = self.mesh.dN_dxyz[element_index, int_point_index]
        node_indices = self.mesh.connectivity[element_index]
        a = jac_N.T @ self.p[node_indices]
        b = jac_N.T @ self.q[node_indices]
        c = numpy.cross(a, b)
        nc = numpy.linalg.norm(c)
        if nc == 0:
            return self.material.C_m()
        else:
            # Compute fiber volume fraction:
            t = c / nc
            v = self.fiber_area * nc
            return self.material.C(v=v, t=t)

    def compliance(
        self,
        p: NDArray,
    ) -> float:
        """
        Compute the structural compliance (twice the strain energy).

        Parameters
        ----------
        p : numpy.ndarray
            Design variables.

        Returns
        -------
        float
            Structural compliance.
        """
        start = clock()

        self.update_p(p)

        compliance = self.u @ self.f

        logger.info(f"Compliance computed in {round(clock() - start, 3)} seconds.")

        return compliance

    def grad_compliance(
        self,
        p: NDArray,
    ) -> numpy.ndarray:
        """
        Compute the gradient of the structural compliance.

        Parameters
        ----------
        p : numpy.ndarray
            Design variables.

        Returns
        -------
        numpy.ndarray
            Gradient of the structural compliance with respect to the
            design variables.
        """
        start = clock()

        self.update_p(p)

        grad = numpy.zeros_like(self.p)

        for element_index, node_indices in enumerate(self.mesh.connectivity):
            dof = utils.dof_indices(node_indices)

            for jac_N, w_det_J in zip(
                self.mesh.dN_dxyz[element_index],
                self.mesh.w_det_dxyz_drst[element_index],
            ):
                # Compute strain:
                strain = utils.B(jac_N) @ self.u[dof]

                # Compute material stiffness matrix derivatives:
                a = jac_N.T @ self.p[node_indices]  # = d p / d xyz
                b = jac_N.T @ self.q[node_indices]  # = d q / d xyz
                c = numpy.cross(a, b)
                nc = numpy.linalg.norm(c)

                if nc == 0:
                    # logger.warning("Zero gradient.")
                    continue

                # Fiber tangent direction:
                t = c / nc

                # Fiber volume fraction:
                v = self.fiber_area * nc

                dc_da = skew(-b)
                # dc_db = skew(a)

                dnc_dc = t.T  # = c / nc (the transpose doesn't matter as it's a 1D array)

                dnc_da = dnc_dc @ dc_da

                dt_dc = (numpy.eye(3) - numpy.outer(t, t)) / nc

                dt_da = dt_dc @ dc_da
                # dt_db = dt_dc @ dc_db

                # d a / d p = jac_N.T
                # fiber_area = d v / d nc
                dv_dp = self.fiber_area * dnc_da @ jac_N.T

                dt_dp = dt_da @ jac_N.T  # (3, 3,) @ (3, nodes per elem) = (3, nodes per elem)

                for node_local, node_global in enumerate(node_indices):
                    grad[node_global] -= (
                        strain.T
                        @ self.material.dC(
                            v=v,
                            t=t,
                            dv=dv_dp[node_local],
                            dt=dt_dp[:, node_local],
                        )
                        @ strain
                        * w_det_J
                    )

        logger.info(f"Compliance gradient computed in {round(clock() - start, 3)} seconds.")

        return grad

    def min_spacing(
        self,
        p: NDArray,
        r: float = -100,
    ) -> tuple[float, NDArray]:
        """
        Compute the approximate minimum fiber spacing and its gradient.

        Parameters
        ----------
        p : numpy.ndarray
            Design variables (scalar field values at nodes).
        r : float
            KS aggregation parameter.

        Returns
        -------
        float
            Approximate minimum fiber spacing.
        numpy.ndarray
            Gradient of the approximate minimum fiber spacing with
            respect to the design variables.
        """
        start = clock()

        if r >= 0:
            raise ValueError("KS aggregation parameters must be negative for minimum aggregation.")

        d = numpy.full(
            shape=self.mesh.n_elements * self.mesh.n_int_points,
            fill_value=numpy.inf,
            dtype=float,
        )

        rows, cols, vals = [], [], []
        for e, n in enumerate(self.mesh.connectivity):
            for i, jac_N in enumerate(self.mesh.dN_dxyz[e]):
                a = jac_N.T @ p[n]  # = d p / d xyz
                b = jac_N.T @ self.q[n]  # = d q / d xyz
                nb = numpy.linalg.norm(b)
                c = numpy.cross(a, b)
                nc = numpy.linalg.norm(c)
                if nc == 0:
                    continue
                d[e * self.mesh.n_int_points + i] = nb / nc
                rows.append(numpy.repeat(e + i, len(n)))
                cols.append(n)
                vals.append(jac_N @ numpy.cross(b, numpy.cross(a, b)) * nb / nc**3)
        jac = scipy.sparse.coo_array(
            (
                numpy.concatenate(vals),
                (
                    numpy.concatenate(rows),
                    numpy.concatenate(cols),
                ),
            ),
            shape=(self.mesh.n_elements * self.mesh.n_int_points, self.mesh.n_nodes),
            dtype=float,
        ).tocsc()

        value, gradient = ks(
            values=d,
            r=r,
            jac=jac,
        )

        logger.info(
            f"Minimum fiber spacing and gradient computed in {round(clock() - start, 3)} seconds."
        )

        return value, gradient

    def total_length(
        self,
        p: NDArray,
    ) -> tuple[float, NDArray]:
        """
        Compute the total fiber length and its gradient.

        Parameters
        ----------
        p : numpy.ndarray
            Design variables (scalar field values at nodes).

        Returns
        -------
        float
            Total fiber length.
        numpy.ndarray
            Gradient of total fiber length with respect to the design
            variables.
        """
        start = clock()

        total_length = 0
        grad_total_length = numpy.zeros_like(p)

        for element, nodes in enumerate(self.mesh.connectivity):
            for jac_N, w_det_J in zip(
                self.mesh.dN_dxyz[element],
                self.mesh.w_det_dxyz_drst[element],
            ):
                a = p[nodes] @ jac_N  # = d q / d xyz
                b = self.q[nodes] @ jac_N  # = d p / d xyz
                c = numpy.cross(a, b)
                nc = numpy.linalg.norm(c)

                if nc == 0:
                    continue

                total_length += nc * w_det_J

                dc_da = skew(-b)
                # dc_db = skew(a)

                dnc_dc = (c / nc).T  # (3,) array, .T only for understanding

                dnc_da = dnc_dc @ dc_da
                # dnc_db = dnc_dc @ dc_db

                # d a / d p = d b / d q = jac_N.T
                dnc_dp = jac_N @ dnc_da
                # dnc_dq = jac_N @ dnc_db

                grad_total_length[nodes] += dnc_dp * w_det_J

        logger.info(
            f"Total fiber length and gradient computed in {round(clock() - start, 3)} seconds."
        )

        return total_length, grad_total_length

    def min_p(
        self,
        p: NDArray,
        r: float = -30,
    ) -> tuple[float, NDArray]:
        """
        Compute the approximate minimum of the design variables.

        Parameters
        ----------
        p : numpy.ndarray
            Design variables (scalar field values at nodes).

        Returns
        -------
        float
            Approximate minimum of the design variables.
        numpy.ndarray
            Gradient of the approximate minimum of the design variables
            with respect to the design variables.
        """
        start = clock()

        value, gradient = ks(
            values=p,
            r=r,
        )

        logger.info(
            f"Soft min. of design variables and its gradient computed "
            f"in {round(clock() - start, 3)} seconds."
        )

        return value, gradient

    def _compute_target_v_and_t(
        self,
        v_min: float,
        v_max: float,
    ) -> tuple[NDArray, NDArray]:
        """Construct target fiber volume fractions and directions at
        integration points.

        Parameters
        ----------
        v_min : float
            Minimum fiber volume fraction.
        v_max : float
            Maximum fiber volume fraction.
        projection_method: oamc.enums.ProjectionMethod, default: oamc.enums.ProjectionMethod.L2
            Method for projecting stress values from integration
            oints to nodes.

        Returns
        -------
        numpy.ndarray
            Target fiber volume fraction as an array of shape (number of
            elements, number of integration points per element).
        numpy.ndarray
            Target fiber direction as an array of shape (number of
            elements, number of integration points per element, 3).
        """
        start = clock()

        n_int_points = self.mesh.n_int_points

        # Allocate memory for major principal stress (mps) directions
        # (dir) and values (val) at integration points:
        mps_dir = numpy.empty(
            shape=(self.mesh.n_elements, n_int_points, 3),
            dtype=float,
        )
        mps_val = numpy.empty(
            shape=(self.mesh.n_elements, n_int_points),
            dtype=float,
        )

        # Compute major principal stress directions and values:
        stress = self.get_stress_at_int_points()
        x = numpy.array([1, 0, 0], dtype=float)
        for e in range(self.mesh.n_elements):
            for i in range(n_int_points):
                (val, dir) = principal_stress(
                    stress_tensor=vector_to_tensor(stress[e, i]),
                    direction=Direction.MAX,
                )
                if dir @ x < 0:
                    dir *= -1
                mps_val[e, i] = val
                mps_dir[e, i] = dir

        # Target fiber volume fraction (v) varies linearly from v_min to
        # v_max between the min. and max. major principal stress (mps)
        # values:
        mps_val = numpy.maximum(mps_val, 0)
        mps_max = numpy.max(mps_val)
        mps_min = numpy.min(mps_val)
        v = v_min + (v_max - v_min) * (mps_val - mps_min) / (mps_max - mps_min)

        logger.info(
            f"Computed target fiber volume fraction and direction in {round(clock() - start, 3)} seconds."
        )

        return v, mps_dir

    def init_p_by_euler_lagrange(
        self,
        v_min: float,
        v_max: float,
        v_average: float,
        iteration_limit: int = 100,
    ) -> None:
        """
        Initialize the design variables such that the fiber
        direction is roughly aligned with the max. principal stress
        in the matrix-only simulation (both scalar fields zero) and
        the fiber density is proportional to the max. principal stress.

        Parameters
        ----------
        v_min : float
            Minimum fiber volume fraction.
        v_max : float
            Maximum fiber volume fraction.s
        v_average : float
            Target average fiber volume fraction.
        iteration_limit : int, default: 100
            Maximum number of iterations.
        projection_method : oamc.enums.ProjectionMethod, default: oamc.enums.ProjectionMethod.L2
            Which method to use to project stress values from
            integration points to nodes.

        Notes
        -----
        The following energy functional is minimized:
        # TODO: Explain energy functional.

        This is done via alternating minimization over the two scalar
        fields p and q.

        Aligning the direction field t is not necessary because it
        only appears as `numpy.outer(t, t)`, which is sign-agnostic.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation
        """

        start = clock()

        # Parameters:
        w1 = 1  # weight for (t x p1)^2
        w2 = 1  # weight for (t x p2)^2
        w3 = 1  # weight for (p1 x p2)^2
        w4 = 5  # weight for (v^2 - (s_max/max(s_max))^2)^2
        epsilon = 1e-3
        convergence_rel_tol = 1e-2
        convergence_abs_tol = 1e-3

        v_target, t_target = self._compute_target_v_and_t(v_min, v_max)

        p = numpy.concatenate((self.mesh.nodes[:, 1], [0]))  # + 1 for zero-mean constraint
        q = numpy.concatenate((self.q, [0]))  # + 1 for zero-mean constraint

        # Precompute zero-mean constraint (zmc) data:
        zmc = numpy.zeros(self.mesh.n_nodes)
        for element_index, node_indices in enumerate(self.mesh.connectivity):
            for int_point, w_det_J in zip(
                utils.INT_POINTS[self.mesh.type],
                self.mesh.w_det_dxyz_drst[element_index],
            ):
                zmc[node_indices] += utils.N(self.mesh.type, int_point) * w_det_J

        zmc_coo_rows = numpy.concatenate(
            (
                numpy.arange(self.mesh.n_nodes),
                numpy.repeat(self.mesh.n_nodes, self.mesh.n_nodes),
            )
        )
        zmc_coo_cols = numpy.concatenate(
            (
                numpy.repeat(self.mesh.n_nodes, self.mesh.n_nodes),
                numpy.arange(self.mesh.n_nodes),
            )
        )
        zmc_coo_vals = numpy.tile(zmc, 2)

        for iteration in range(iteration_limit):
            K_rows_1, K_cols_1, K_vals_1 = [], [], []
            # K_rows_2, K_cols_2, K_vals_2 = [], [], []
            f_1 = numpy.zeros(self.mesh.n_nodes + 1)  # + 1 for zero-mean constraint
            # f_2 = numpy.zeros(self.mesh.node_count + 1)  # + 1 for zero-mean constraint
            for element_index, node_indices in enumerate(self.mesh.connectivity):
                K_e_1 = numpy.zeros((node_indices.size, node_indices.size))
                # K_e_2 = numpy.zeros((node_indices.size, node_indices.size))
                for t_target_i, v_target_i, jac_N, w_det_J in zip(
                    t_target[element_index],
                    v_target[element_index],
                    self.mesh.dN_dxyz[element_index],
                    self.mesh.w_det_dxyz_drst[element_index],
                ):
                    a = jac_N.T @ p[node_indices]
                    b = jac_N.T @ q[node_indices]
                    c = numpy.cross(a, b)
                    # na = numpy.linalg.norm(a)
                    nb = numpy.linalg.norm(b)
                    nc = numpy.linalg.norm(c)

                    # Residual of squared fiber volume fraction:
                    r = (nc * self.fiber_area) ** 2 - v_target_i**2
                    dr_da = 2 * (nb**2 * a - numpy.dot(a, b) * b) * (self.fiber_area) ** 2
                    # dr_db = 2 * (na**2 * b - numpy.dot(a, b) * a) * (self.fiber_area) ** 2

                    A_1 = (
                        w1 * numpy.outer(t_target_i, t_target_i)
                        + w3 * numpy.outer(b, b)
                        + w4 * numpy.outer(dr_da, dr_da)
                        + epsilon * numpy.eye(3)
                    )
                    # A_2 = (
                    #     w2 * numpy.outer(mps_dir_i, mps_dir_i)
                    #     + w3 * numpy.outer(a, a)
                    #     + w4 * numpy.outer(dr_db, dr_db)
                    #     + epsilon * numpy.eye(3)
                    # )

                    K_e_1 += jac_N @ A_1 @ jac_N.T * w_det_J
                    # K_e_2 += jac_N_i @ A_2 @ jac_N_i.T * w_det_J_i

                    f_1[node_indices] -= jac_N @ (w4 * dr_da * r) * w_det_J
                    # f_2[node_indices] -= jac_N_i @ (w4 * dr_db * r) * w_det_J_i

                    K_rows_1.append(numpy.repeat(node_indices, node_indices.size))
                    K_cols_1.append(numpy.tile(node_indices, node_indices.size))
                    K_vals_1.append(K_e_1.ravel())
                    # K_rows_2.append(numpy.repeat(node_indices, node_indices.size))
                    # K_cols_2.append(numpy.tile(node_indices, node_indices.size))
                    # K_vals_2.append(K_e_2.ravel())

            K_1 = scipy.sparse.coo_array(
                arg1=(
                    numpy.concatenate(K_vals_1 + [zmc_coo_vals]),
                    (
                        numpy.concatenate(K_rows_1 + [zmc_coo_rows]),
                        numpy.concatenate(K_cols_1 + [zmc_coo_cols]),
                    ),
                ),
                shape=(self.mesh.n_nodes + 1, self.mesh.n_nodes + 1),
            ).tocsr()
            # K_2 = scipy.sparse.coo_array(
            #     arg1=(
            #         numpy.concatenate(K_vals_2 + [zmc_coo_vals]),
            #         (
            #             numpy.concatenate(K_rows_2 + [zmc_coo_rows]),
            #             numpy.concatenate(K_cols_2 + [zmc_coo_cols]),
            #         ),
            #     ),
            #     shape=(self.mesh.node_count + 1, self.mesh.node_count + 1),
            # ).tocsr()

            K_1.sort_indices()
            # K_2.sort_indices()

            p_1_old = p.copy()
            p_2_old = q.copy()

            p = pypardiso.spsolve(K_1, f_1)
            # q = pypardiso.spsolve(K_2, f_2)

            # Check convergence:
            if numpy.allclose(
                a=p,
                b=p_1_old,
                rtol=convergence_rel_tol,
                atol=convergence_abs_tol,
            ) and numpy.allclose(
                a=q,
                b=p_2_old,
                rtol=convergence_rel_tol,
                atol=convergence_abs_tol,
            ):
                logger.info(
                    f"Scalar field initialization converged in {round(clock() - start, 3)} "
                    f"seconds after {iteration + 1} iteration(s)."
                )
                break
            elif iteration == iteration_limit - 1:
                logger.info(
                    f"Scalar field initialization reached iteration limit ({iteration_limit}) "
                    f"in {round(clock() - start, 3)} seconds."
                )
                break

        # Scale the scalar fields to achieve the target average fiber volume fraction:
        # weighted_sum_of_v = 0
        # for element_index, node_indices in enumerate(self.mesh.connectivity):
        #     for jac_N, w_det_J in zip(
        #         self.mesh.dN_dxyz[element_index],
        #         self.mesh.w_det_dxyz_drst[element_index],
        #     ):
        #         a = jac_N.T @ p[node_indices]
        #         b = jac_N.T @ q[node_indices]
        #         c = numpy.cross(a, b)
        #         nc = numpy.linalg.norm(c)
        #         if nc == 0:
        #             continue
        #         v = self.fiber_area * nc
        #         weighted_sum_of_v += v * w_det_J
        # v_average = weighted_sum_of_v / numpy.sum(self.mesh.w_det_dxyz_drst)
        # p_1 *= numpy.sqrt(target_v_average / v_average)
        # p_2 *= numpy.sqrt(target_v_average / v_average)

        self.p = p[: self.mesh.n_nodes]

        if numpy.isnan(self.p).any():
            logger.critical("NaN detected in p.")
        if numpy.isinf(self.p).any():
            logger.critical("Inf detected in p.")

    def init_p_by_least_squares(
        self,
        v_min: float,
        v_max: float,
    ) -> None:
        start = clock()

        v_target, t_target = self._compute_target_v_and_t(v_min, v_max)
        v_target = v_target.reshape(-1)
        t_target = t_target.reshape(-1, 3)

        c_target = (v_target / self.fiber_area)[:, None] * t_target

        def r(p: NDArray) -> NDArray:
            # Allocate momory for the cross products:
            c = numpy.empty(
                shape=(self.mesh.n_elements * self.mesh.n_int_points, 3),
                dtype=float,
            )

            # Compute the cross products:
            for e, n in enumerate(self.mesh.connectivity):
                for i, jac_N in enumerate(self.mesh.dN_dxyz[e]):
                    a = jac_N.T @ p[n]
                    b = jac_N.T @ self.q[n]
                    c[e * self.mesh.n_int_points + i] = numpy.cross(a, b)

            # Compute the residuals:
            r = (numpy.sqrt(self.mesh.w_det_dxyz_drst.ravel())[:, None] * (c - c_target)).ravel()

            # Compute the Jacobian of the residuals at integration points with respect to p at
            # nodes:
            rows, cols, vals = [], [], []
            for e, n in enumerate(self.mesh.connectivity):
                for i, (jac_N, w_det_J) in enumerate(
                    zip(
                        self.mesh.dN_dxyz[e],
                        self.mesh.w_det_dxyz_drst[e],
                    )
                ):
                    b = jac_N.T @ self.q[n]
                    # Transpose of the Jacobian of the residual vector at the current integration
                    # point with respect to p at the nodes of the current element:
                    dr_dp = numpy.sqrt(w_det_J) * numpy.cross(jac_N, b)
                    # Shape: (number of nodes per element, 3)

                    rows.append(
                        numpy.tile(
                            numpy.repeat(3 * (e * self.mesh.n_int_points + i), 3)
                            + numpy.arange(3),
                            len(n),
                        )
                    )
                    cols.append(numpy.repeat(n, 3))
                    vals.append(dr_dp.ravel())

            jac_r = scipy.sparse.coo_array(
                (
                    numpy.concatenate(vals),
                    (
                        numpy.concatenate(rows),
                        numpy.concatenate(cols),
                    ),
                ),
                shape=(self.mesh.n_elements * self.mesh.n_int_points * 3, self.mesh.n_nodes),
            ).tocsr()

            return r, jac_r

        r_cache = FunctionCache(fun_and_jac=r)

        result: scipy.optimize.OptimizeResult = scipy.optimize.least_squares(
            fun=r_cache.fun,
            x0=self.p,
            jac=r_cache.jac,
        )

        logger.info(
            f"p initialized using least squares after {result.nfev} "
            f"function and {result.njev} Jacobian evaluations in "
            f"{round(clock() - start, 3)} seconds."
        )

        self.p = result.x

    def init_q(self) -> None:
        start = clock()

        self.q = self.get_distance_to_mold(self.mesh.nodes)
        self.q /= self.layer_height
        self.q += 0.5

        logger.info(f"q initialized in {round(clock() - start, 3)} seconds.")

    def optimize_p(
        self,
        max_fiber_length: float | None = None,
        max_fiber_volume_fraction: float | None = None,
        max_fiber_weight: float | None = None,
        callback: Callable[[scipy.optimize.OptimizeResult], None] | None = None,
        iteration_limit: int = 100,
    ) -> None:
        """Optimize fiber trajectories for minimum structural compliance.

        Parameters
        ----------
        min_spacing : float
            Minimum allowed fiber spacing.
        max_fiber_length : float, optional
            Maximum allowed fiber length.
        max_fiber_volume_fraction : float, optional
            Maximum allowed fiber volume fraction.
        max_fiber_weight : float, optional
            Maximum allowed fiber weight.
        """

        start = clock()

        if max_fiber_length is not None and max_fiber_length <= 0:
            raise ValueError("Maximum fiber length must be positive.")
        if max_fiber_volume_fraction is not None:
            raise NotImplementedError("Fiber volume fraction constraint is not yet implemented.")
        if max_fiber_weight is not None:
            raise NotImplementedError("Fiber weight constraint is not yet implemented.")

        min_spacing_cache = FunctionCache(self.min_spacing)
        total_length_cache = FunctionCache(self.total_length)
        min_p_cache = FunctionCache(self.min_p)

        fiber_width = self.fiber_area / self.layer_height

        result: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            fun=self.compliance,
            jac=self.grad_compliance,
            x0=self.p,
            method="trust-constr",
            constraints=[
                scipy.optimize.NonlinearConstraint(
                    fun=min_spacing_cache.fun,
                    lb=fiber_width,
                    ub=numpy.inf,
                    jac=min_spacing_cache.jac,
                ),
                scipy.optimize.NonlinearConstraint(
                    fun=total_length_cache.fun,
                    lb=-numpy.inf,
                    ub=max_fiber_length,
                    jac=total_length_cache.jac,
                ),
                # scipy.optimize.NonlinearConstraint(
                #     fun=min_p_cache.fun,
                #     lb=0,
                #     ub=0,
                #     jac=min_p_cache.jac,
                # ),
            ],
            callback=callback,
            options={
                "disp": True,
                "maxiter": iteration_limit,
            },
        )

        logger.info(
            f"Optimization completed in {round(clock() - start, 3)} seconds: {result.message}"
        )

        self.p = result.x

    def filter_p_by_diffusion(self, diffusion_level: float, iterations: int = 1) -> None:
        """Filter p to prevent small kinks in the fiber paths, for
        example.

        # TODO: Explain how this is related to real diffusion.

        Parameters
        ----------
        level : float, default = 1
            Level of diffusion. In a real diffusion problem, this
            would be proportional to the time step.
        iterations : int, default: 1
            Number of iterations to perform.
        """
        start = clock()

        M_rows, M_columns, M_values = [], [], []
        K_rows, K_columns, K_values = [], [], []
        for element, nodes in enumerate(self.mesh.connectivity):
            # Compute the elemental mass and stiffness matrices:
            M_e = numpy.zeros((nodes.size, nodes.size))
            K_e = numpy.zeros((nodes.size, nodes.size))
            for N, jac_N, w_det_J in zip(
                self.mesh.N,
                self.mesh.dN_dxyz[element],
                self.mesh.w_det_dxyz_drst[element],
            ):
                M_e += numpy.outer(N, N) * w_det_J
                K_e += jac_N @ jac_N.T * w_det_J

            # Add to global mass matrix:
            M_rows.append(numpy.repeat(nodes, nodes.size))
            M_columns.append(numpy.tile(nodes, nodes.size))
            M_values.append(K_e.ravel())
            # Add to global mass matrix:
            K_rows.append(numpy.repeat(nodes, nodes.size))
            K_columns.append(numpy.tile(nodes, nodes.size))
            K_values.append(K_e.ravel())

        # Assemble the global mass and stiffness matrices in COO format,
        # then convert to CSR format for solving:
        M = scipy.sparse.coo_array(
            arg1=(
                numpy.concatenate(M_values),
                (
                    numpy.concatenate(M_rows),
                    numpy.concatenate(M_columns),
                ),
            ),
            shape=(self.mesh.n_nodes, self.mesh.n_nodes),
        ).tocsr()
        M.sort_indices()
        K = scipy.sparse.coo_array(
            arg1=(
                numpy.concatenate(K_values),
                (
                    numpy.concatenate(K_rows),
                    numpy.concatenate(K_columns),
                ),
            ),
            shape=(self.mesh.n_nodes, self.mesh.n_nodes),
        ).tocsr()
        K.sort_indices()

        average_element_size = numpy.cbrt(self.mesh.volume / self.mesh.n_elements)
        dt = diffusion_level * average_element_size**2 / iterations

        for _ in range(iterations):
            self.p = pypardiso.spsolve(M + dt * K, M @ self.p)

        logger.info(f"p filtered by diffusion in {round(clock() - start, 3)} seconds.")

    def compute_fibers(
        self,
        p_splits: int = 1,
        min_length: float = 0,
    ) -> dict[int, list[Fiber]]:
        """
        Compute fiber paths as intersection curves of the level
        surfaces of the two scalar fields (design variables).

        Parameters
        ----------
        p_splits : int
            Introduce intermediate p-isosurfaces. Adjacent q-isosurface
            will be paired with different sets of p-isosurfaces in order
            to obtain more evenly distributed fiber paths.
        min_length : float, default: 0
            Fibers shorter than this length will not be saved.

        Returns
        -------
        dict of int to list of oamc.path.Path
            Fiber paths sorted by layer.
        """
        start = clock()

        lists_of_points = compute_int_isosurface_intersections(
            self.get_grid(),
            self.p,
            self.q,
            p_splits=p_splits,
        )

        fibers = defaultdict(list)
        height = self.layer_height
        width = self.fiber_area / height

        for (p, q), list_of_points in lists_of_points.items():
            for points in list_of_points:
                normals = self.mold.get_unit_normal_vectors(points)
                fiber = Fiber(
                    points=points,
                    orientations=normals,
                    dims=(width, height),
                )
                if fiber.length >= min_length:
                    fibers[int(round(q))].append(fiber)

        logger.info(f"Fiber paths computed in {round(clock() - start, 3)} seconds.")

        self.fibers = fibers

        return fibers

    def downsample_fibers_by_rdp(self, max_deviation: float) -> None:
        for fiber in self.fibers_as_list:
            fiber.downsample_by_rdp(max_deviation)

    def downsample_fibers_by_vw(self, min_area: float) -> None:
        for fiber in self.fibers_as_list:
            fiber.downsample_by_vw(min_area)

    def remove_outliers(self, max_length: float, min_angle: float) -> None:
        for fiber in self.fibers_as_list:
            fiber.remove_outliers(max_length, min_angle)

    def mirror_fibers(self, point: NDArray, normal: NDArray) -> None:
        """Mirror all fibers about the given mirror plane.

        Parameters
        ----------
        point : numpy.ndarray
            An arbitrary point on the mirror plane.
        normal : numpy.ndarray
            An arbitrary vector normal to the mirror plane.
        """

        raise NotImplementedError("Mirroring fibers is not yet implemented.")

        # TODO: Implement CompositeModel.mirror_fibers.

    @property
    def fibers_as_list(self) -> list[Fiber]:
        return list(chain.from_iterable(self.fibers.values()))

    @property
    def total_number_of_points(self) -> int:
        n = 0
        for fiber in self.fibers_as_list:
            n += fiber.points.shape[0]
        return n

    @property
    def precise_total_length(self) -> float:
        length = 0
        for fibers in self.fibers.values():
            for fiber in fibers:
                length += fiber.length
        return length

    def read_fibers(self, directory: str):
        # TODO: Implement oamc.x.composite_model.CompositeModel.read_fibers
        ...

    def save_fibers(
        self,
        directory: str,
        subdirectories: bool = False,
        convention: Literal["unit vector", "XYZ"] = "XYZ",
        angle_unit: AngleUnit = AngleUnit.RAD,
        delimiter: str = ",",
        decimals: int = 5,
    ) -> None:
        """Save all fiber paths.

        Parameters
        ----------
        directory : str
            Directory where the fibers will be saved.
        subdirectories : bool, default: False
            Whether to make one subdirectory per layer.
        convention : {"XYZ"}
            Euler angle convention (see [1]_). Capital letters indicate global axes.
        angle_unit : {"rad", "deg"}
            Unit of the exported angles.
        delimiter : str, default: ","
            Delimiter to use in the CSV file.
        decimals : int, default: 3
            Number of decimal places to save.
        """
        start = clock()

        if self.fibers is None:
            logger.warning("No fibers to save. Compute fibers before saving.")
            return

        directory = Path(directory).resolve()
        makedirs(directory, exist_ok=True)

        global_i = 0
        for layer, fibers in self.fibers.items():
            for i, fiber in enumerate(fibers):
                if subdirectories:
                    subdirectory = directory / f"layer_{layer}"
                    makedirs(subdirectory, exist_ok=True)
                    path = subdirectory / f"fiber_{layer}_{i + 1}.csv"
                else:
                    path = directory / f"fiber_{global_i + 1}.csv"
                fiber.save(
                    path=path,
                    convention=convention,
                    angle_unit=angle_unit,
                    delimiter=delimiter,
                    decimals=decimals,
                )
                global_i += 1

        logger.info(f"Fiber paths saved in {round(clock() - start, 3)} seconds.")
