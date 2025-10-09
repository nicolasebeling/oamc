import numpy
from numpy.typing import NDArray

from oamc.fea.material import IsotropicMaterial, Material, TransverselyIsotropicMaterial
from oamc.x import utils as utils


class CompositeMaterial(Material):
    def __init__(
        self,
        matrix_material: IsotropicMaterial,
        fiber_material: TransverselyIsotropicMaterial,
        fiber_diameter: float,
    ):
        self.matrix_material = matrix_material
        self.fiber_material = fiber_material

        C = fiber_material.C
        self.c1 = C[1, 2]
        self.c2 = C[3, 3]
        self.c3 = C[0, 1] - C[1, 2]
        self.c4 = 2 * C[4, 4] - 2 * C[3, 3]
        self.c5 = C[0, 0] - 2 * C[0, 1] + C[1, 2] + 2 * C[3, 3] - 4 * C[4, 4]

        if fiber_diameter <= 0:
            raise ValueError("Fiber diameter must be positive.")
        self.fiber_diameter = fiber_diameter

    def C_m(self) -> NDArray:
        """
        :return C_m: matrix material stiffness matrix
        """
        return self.matrix_material.C

    def C_f(self, t: NDArray) -> NDArray:
        """
        :param v: fiber volume fraction
        :param t: fiber tangent direction as a 3D unit vector
        :return C_f: fiber material stiffness matrix
        """

        if not numpy.isclose(numpy.linalg.norm(t), 1.0):
            raise ValueError("Fiber direction t must be a unit vector.")

        T = numpy.outer(t, t)

        I = numpy.eye(3)

        Is = numpy.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        Is[i, j, k, l] = 0.5 * (int(i == k and j == l) + int(i == l and j == k))

        C_f = (
            self.c1 * utils.dyadic_product(I, I)
            + 2.0 * self.c2 * Is
            + self.c3 * (utils.dyadic_product(T, I) + utils.dyadic_product(I, T))
            + self.c4 * (utils.bar_product(T, I) + utils.bar_product(I, T))
            + self.c5 * utils.dyadic_product(T, T)
        )

        # Enforce minor symmetries to prevent numerical drift:
        C_f = 0.5 * (C_f + C_f.transpose(0, 1, 3, 2))
        C_f = 0.5 * (C_f + C_f.transpose(1, 0, 2, 3))

        return utils.tensor_to_matrix(C_f)

    def C(self, v: float, t: NDArray) -> NDArray:
        """
        :param v: fiber volume fraction
        :param t: fiber tangent direction
        :return C: material stiffness matrix
        """

        if v < 0 or v > 1:
            raise ValueError("Fiber volume fraction must be in [0, 1].")

        return self.C_f(t) * v + self.C_m() * (1 - v)

    def S(self, v: NDArray, t: NDArray) -> NDArray:
        return numpy.linalg.inv(self.C(v=v, t=t))

    def apply_dC_dv(self, t: NDArray, dv: NDArray) -> NDArray:
        """
        :param v: fiber volume fraction
        :param t: fiber tangent direction as a 3D unit vector
        :param dv: increment in fiber volume fraction
        :return dC: increment in material stiffness
        """
        return (self.C_f(t) - self.C_m()) * dv

    def apply_dC_dt(self, v: float, t: NDArray, dt: NDArray) -> NDArray:
        """
        :param v: fiber volume fraction
        :param t: fiber tangent direction as a 3D unit vector
        :param dt: increment in fiber tangent direction
        :return dC: increment in material stiffness
        """

        if not numpy.isclose(numpy.linalg.norm(t), 1.0):
            raise ValueError("Fiber direction t must be a unit vector.")

        T = numpy.outer(t, t)

        I = numpy.eye(3)

        # Project dt to be perpendicular to t to maintain unit length of t:
        P = numpy.eye(3) - T  # projector
        dt = P @ dt

        # Product rule:
        dT = numpy.outer(dt, t) + numpy.outer(t, dt)

        dC = (
            self.c3 * (utils.dyadic_product(dT, I) + utils.dyadic_product(I, dT))
            + self.c4 * (utils.bar_product(dT, I) + utils.bar_product(I, dT))
            + self.c5 * (utils.dyadic_product(dT, T) + utils.dyadic_product(T, dT))
        ) * v

        # Enforce minor symmetries to prevent numerical drift:
        dC = 0.5 * (dC + dC.transpose(0, 1, 3, 2))
        dC = 0.5 * (dC + dC.transpose(1, 0, 2, 3))

        return utils.tensor_to_matrix(dC)

    def dC(self, v: float, t: NDArray, dv: NDArray, dt: NDArray) -> NDArray:
        """
        :param v: fiber volume fraction
        :param t: fiber tangent direction as a 3D unit vector
        :param dv: increment in fiber volume fraction
        :param dt: increment in fiber tangent direction
        :return dC: increment in material stiffness
        """
        return self.apply_dC_dv(t, dv) + self.apply_dC_dt(v, t, dt)


if __name__ == "__main__":
    # TODO: Make this a test.

    # Generate a random rotation matrix:
    R, _ = numpy.linalg.qr(numpy.random.randn(3, 3))
    if numpy.linalg.det(R) < 0:
        R[:, 0] *= -1

    # Fiber tangent direction = first column of rotation matrix (passive convention):
    t = R[:, 0]

    matrix_material = IsotropicMaterial(
        E=1e9,
        nu=0.35,
        rho=1000,
    )

    fiber_material = TransverselyIsotropicMaterial(
        E1=130e9,
        E2=8e9,
        nu12=0.25,
        G23=3e9,
        G12=4e9,
        rho=1750,
    )

    composite_material = CompositeMaterial(
        matrix_material=matrix_material,
        fiber_material=fiber_material,
        fiber_diameter=1.0,
    )

    c_analytic = numpy.array(
        [
            composite_material.c1,
            composite_material.c2,
            composite_material.c3,
            composite_material.c4,
            composite_material.c5,
        ]
    )

    # Fiber volume fraction:
    v = 0.5

    C1 = fiber_material.C_transformed(R) * v + matrix_material.C_transformed(R) * (1 - v)
    C2 = composite_material.C(v=v, t=t)

    print(f"Max. rel. error = {numpy.max(numpy.abs((C2.ravel() - C1.ravel()) / C1.ravel()))}")
