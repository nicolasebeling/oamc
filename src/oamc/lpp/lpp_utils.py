import logging
from typing import Callable

from numpy.typing import NDArray

from oamc import mechanics_utils
from oamc.enums import Direction, RKMethod

logger = logging.getLogger(__name__)


def pointing_stress_vector(stress_vector: NDArray, direction: Direction) -> NDArray:
    """Compute the pointing stress vector as defined in [1].

    NOTE: The pointing stress vectors are not yet normalized.

    :param stress: stress vector
    :param axis: axis along which the stress vector shall be computed
    :return: pointing stress vector
    """
    stress_tensor = mechanics_utils.vector_to_tensor(stress_vector)
    match direction:
        case Direction.X:
            return stress_tensor[:, 0]
        case Direction.Y:
            return stress_tensor[:, 1]
        case Direction.Z:
            return stress_tensor[:, 2]
        case Direction.MIN | Direction.INT | Direction.MAX:
            value, vector = mechanics_utils.principal_stress(
                stress_tensor=stress_tensor,
                direction=direction,
            )
            return vector * value
        case _:
            raise ValueError(f"Unknown axis: {direction}")


def integration_step(
    f: Callable[[NDArray], NDArray],
    x: NDArray,
    step_size: float,
    direction: NDArray | None = None,
    method: RKMethod = RKMethod.RK4,
) -> NDArray:
    """Compute one Runge-Kutta (RK) integration step.

    :param f: function to integrate
    :param x: current point
    :param step_size: step size for the RK4 method
    :param direction: The direction is chosen such that the dot product of the
        provided direction and the current step is positive. This is useful
        when integrating eigenvectors, which may randomly change direction.
    :param method: RK integration scheme
    :return: RK step
    """

    match method:
        case RKMethod.RK4:
            k1 = f(x)
            if direction is not None and direction @ k1 < 0:
                k1 = -k1
            k2 = f(x + step_size * k1 / 2)
            if direction is not None and direction @ k2 < 0:
                k2 = -k2
            k3 = f(x + step_size * k2 / 2)
            if direction is not None and direction @ k3 < 0:
                k3 = -k3
            k4 = f(x + k3)
            if direction is not None and direction @ k4 < 0:
                k4 = -k4
            return step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        case _:
            raise ValueError(f"Unknown RK method: {method}")
