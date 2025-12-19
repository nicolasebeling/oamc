"""Utilities for working with polylines."""

import numpy
from numpy.typing import NDArray


def distance_to_line(a: NDArray, b: NDArray, c: NDArray) -> float:
    """Compute the distance of a point from a straight line.

    Parameters
    ----------
    a : numpy.ndarray
        First endpoint of the line.
    b : numpy.ndarray
        Second endpoint of the line.
    c : numpy.ndarray
        Point whose distance to the line shall be computed.

    Returns
    -------
    float
        Distance of point `c` to the straight line between `a` and `b`.
    """
    u = b - a
    v = c - a
    return numpy.linalg.norm(numpy.cross(u, v)) / numpy.linalg.norm(u)


def rdp(
    polyline: NDArray,
    max_deviation: float,
    return_mask: bool = False,
) -> NDArray:
    """Ramer-Douglas-Peucker (RDP) algorithm for downsampling polylines.

    Parameters
    ----------
    polyline : numpy.ndarray
        Polyline to be downsampled as an array of shape (number of
        points, number of coordinates per point).
    max_deviation : float
        Maximum distance/deviation from the original polyline.
    return_mask : bool, default: False
        Whether to return the mask applied to the original polyline
        (0 where removed, 1 where kept).

    Returns
    -------
    numpy.ndarray
        Downsampled polyline.

    References
    ----------
    .. [1] Wikipedia Contributors, "Ramer-Douglas-Peucker algorithm," Wikipedia, Sep. 01, 2022. https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
    """
    if max_deviation == 0:
        return polyline

    i_max = 0
    d_max = 0
    for i in range(1, len(polyline) - 1):
        d = distance_to_line(polyline[0], polyline[-1], polyline[i])
        if d > d_max:
            i_max = i
            d_max = d

    if d_max > max_deviation:
        # Recursive function calls:
        if return_mask:
            l1, m1 = rdp(polyline[: i_max + 1], max_deviation, return_mask)
            l2, m2 = rdp(polyline[i_max:], max_deviation, return_mask)
            return numpy.vstack((l1[:-1], l2)), numpy.hstack((m1[:-1], m2)).astype(bool)
        else:
            l1 = rdp(polyline[: i_max + 1], max_deviation, return_mask)
            l2 = rdp(polyline[i_max:], max_deviation, return_mask)
            return numpy.vstack((l1[:-1], l2))
    else:
        # End of recursion:
        if return_mask:
            return (
                numpy.array((polyline[0], polyline[-1])),
                numpy.hstack(([1], (len(polyline) - 2) * [0], [1])).astype(bool),
            )
        else:
            return numpy.array((polyline[0], polyline[-1]))


def vw(
    polyline: NDArray,
    min_area: float,
    return_mask: bool = False,
) -> NDArray:
    """Visvalingam-Whyatt (VW) algorithm for downsampling polylines.

    Parameters
    ----------
    polyline : numpy.ndarray
        Polyline to be downsampled as an array of shape (number of
        points, number of coordinates per point).
    min_area : float
        Minimum area of the triangle `(polyline[i - 1], polyline[i],
        polyline[i + 1])` for `polyline[i]` to be kept.
    return_mask : bool, default: False
        Whether to return the mask applied to the original polyline
        (0 where removed, 1 where kept).

    Returns
    -------
    numpy.ndarray
        Downsampled polyline.

    References
    ----------
    .. [1] Wikipedia Contributors, "Visvalingam-Whyatt algorithm," Wikipedia, May 31, 2024. https://en.wikipedia.org/wiki/Visvalingam%E2%80%93Whyatt_algorithm
    """
    mask = numpy.ones(len(polyline), dtype=bool)

    for i in range(1, len(polyline) - 1):
        v = polyline[i - 1] - polyline[i]
        w = polyline[i + 1] - polyline[i]
        if numpy.linalg.norm(numpy.cross(v, w)) / 2 < min_area:
            mask[i] = False

    if return_mask:
        return polyline[mask], mask
    else:
        return polyline[mask]


def remove_outliers(
    polyline: NDArray,
    max_length: float,
    min_angle: float,
    return_mask: bool = False,
) -> NDArray:
    """Downsample a polyline by removing outlier points where the angle
    between consecutive polyline segments shorter than the given maximum
    length exceeds the given minimum angle.

    Parameters
    ----------
    polyline : numpy.ndarray
        Polyline to be downsampled as an array of shape (number of
        points, number of coordinates per point).
    max_length : float
        Maximum length of one of the segments `(polyline[i - 1],
        polyline[i])` and `(polyline[i], polyline[i + 1])` for
        `polyline[i]` to be removed.
    min_angle : float
        Minimum angle `polyline[i - 1]` - `polyline[i]` -
        `polyline[i + 1]` for `polyline[i]` to be removed.
    return_mask : bool, default: False
        Whether to return the mask applied to the original polyline
        (0 where removed, 1 where kept).

    Returns
    -------
    numpy.ndarray
        Downsampled polyline.
    """
    mask = numpy.ones(shape=len(polyline), dtype=bool)

    min_cos = numpy.cos(min_angle)
    for i in range(1, len(polyline) - 1):
        v = polyline[i] - polyline[i - 1]
        w = polyline[i + 1] - polyline[i]
        nv = numpy.linalg.norm(v)
        nw = numpy.linalg.norm(w)
        if nv == 0 or nw == 0:
            continue
        cos = numpy.dot(v, w) / nv / nw
        if cos < min_cos and (nv < max_length or nw < max_length):
            mask[i] = False

    if return_mask:
        return polyline[mask], mask
    else:
        return polyline[mask]


def resample(polyline: NDArray, spacing: float) -> NDArray:
    """

    Parameters
    ----------
    polyline : numpy.ndarray
        Polyline to be downsampled as an array of shape (number of
        points, number of coordinates per point).
    spacing : float
        ...

    Returns
    -------
    numpy.ndarray
        Resampled polyline.
    """

    # TODO: Implement resampling algorithm.


def mirror_polyline_about_plane(
    polyline: numpy.ndarray,
    plane_point: numpy.ndarray,
    plane_normal: numpy.ndarray,
    tol: float = 1e-6,
):
    """Mirror a polyline about a plane defined by a point and a normal vector.

    Parameters
    ----------
    polyline : numpy.ndarray
        Polyline to be mirrored as an array of shape (N, 3).
    plane_point : numpy.ndarray
        An arbitrary point on the mirror plane.
    plane_normal : numpy.ndarray
        An arbitrary vector normal to the mirror plane.
    tol : float
        Tolerance for connecting endpoints.

    Returns
    -------
    tuple of numpy.ndarray
        Shape (1,) if merged, otherwise shape (2,).
    """

    # Normalize the normal vector:
    norm = numpy.linalg.norm(plane_normal)
    if norm == 0:
        raise ValueError("plane_normal must not be the zero vector.")
    plane_normal /= norm

    # Mirror each point p across the plane:
    # d = (p - p0) * n_hat
    # p' = p - 2 d n_hat
    d = numpy.dot(polyline - plane_point, plane_normal)
    mirrored = polyline - 2 * d[:, None] * plane_normal[None, :]

    # Try to connect original and mirrored polylines at endpoints:

    mirrored_rev = mirrored[::-1]

    # Build candidates as (distance, primary, secondary).
    # Always connect primary[-1] to secondary[0].
    candidates = [
        (
            numpy.linalg.norm(polyline[-1] - mirrored[0]),
            polyline,
            mirrored,
        ),
        (
            numpy.linalg.norm(polyline[-1] - mirrored[-1]),
            polyline,
            mirrored_rev,
        ),
        (
            numpy.linalg.norm(polyline[0] - mirrored[0]),
            mirrored_rev,
            polyline,
        ),
        (
            numpy.linalg.norm(polyline[0] - mirrored[-1]),
            mirrored,
            polyline,
        ),
    ]

    # Select best connection:
    distances = [c[0] for c in candidates]
    best_index = numpy.argmin(distances)
    best_distance, primary, secondary = candidates[best_index]

    if best_distance <= tol:
        # Merge, omitting the duplicated connection point in the secondary:
        merged = numpy.vstack([primary, secondary[1:]])

        # Optionally close loop if start and end are within tolerance:
        if numpy.linalg.norm(merged[0] - merged[-1]) <= tol:
            if not numpy.allclose(merged[0], merged[-1]):
                merged = numpy.vstack([merged, merged[0]])

        return (merged,)
    else:
        # No connection, return original and mirrored (in original order):
        return (polyline, mirrored)
