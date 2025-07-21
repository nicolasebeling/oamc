import numpy
from numpy.typing import NDArray


def von_mises_stress(s: NDArray) -> NDArray:
    """
    Calculates the equivalent tensile (von Mises) stress.

    :param s: N x 6 stress vectors in Ansys format [[X, Y, Z, XY, YZ, ZX], ...]
    :return: N equivalent tensile (von Mises) stresses
    """
    if s.shape[1] != 6:
        message = "Incorrect stress vector format. Correct format: [[X, Y, Z, XY, YZ, ZX], ...]"
        raise ValueError(message)

    temp1 = ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) / 2
    temp2 = 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    return numpy.sqrt(temp1 + temp2)


def convert_to_tensor(vector: NDArray):
    return numpy.array(
        [
            [vector[0], vector[3], vector[5]],
            [vector[3], vector[1], vector[4]],
            [vector[5], vector[4], vector[2]],
        ]
    )


def convert_to_vector(tensor: NDArray):
    return NDArray(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            tensor[0, 1],
            tensor[1, 2],
            tensor[2, 0],
        ]
    )
