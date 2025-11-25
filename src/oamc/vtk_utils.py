"""Utility functions for vtk and pyvista."""

from collections import defaultdict

import numpy
import pyvista
import vtk
from numpy.typing import NDArray
from scipy.spatial import KDTree


def compute_level_surface(
    grid: pyvista.UnstructuredGrid,
    name: str,
    level: float,
) -> pyvista.PolyData:
    """
    Compute a single level surface.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        Grid.
    name : str
        Name of the scalar field to contour.
    level : float
        Level value.

    Returns
    -------
    pyvista.PolyData
        Level surface.
    """
    # PyVista will internally pick an appropriate VTK contour filter.
    # Using a single level here makes it easier to organize the results.
    surface: pyvista.PolyData = grid.contour(isosurfaces=[float(level)], scalars=name)
    if surface.n_points:
        # Some VTK filters may leave degenerate polys. Topological
        # cleanup without moving points:
        surface = surface.clean(tolerance=0.0)
    return surface


def convert_to_triangle_mesh(
    polydata: pyvista.PolyData,
    cleaning_tolerance: float | None = None,
) -> pyvista.PolyData:
    """Clean a PolyData surface and convert it to a triangle mesh in
    order to improve the robustness of subsequent geometric operations.

    This function removes duplicate or nearly coincident points and converts
    all polygonal faces to triangles. The resulting mesh is topologically
    simpler and better suited for downstream algorithms such as surface
    intersection or contouring.

    Parameters
    ----------
    polydata : pyvista.PolyData
        Input surface mesh to clean and triangulate.

    Returns
    -------
    pyvista.PolyData
        Cleaned and fully triangulated surface mesh. If the input contains
        no points, it is returned unchanged.
    """

    if polydata.n_points == 0:
        return polydata

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.PointMergingOn()

    if cleaning_tolerance is not None:
        cleaner.SetTolerance(cleaning_tolerance)
        cleaner.ToleranceIsAbsoluteOn()

    cleaner.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cleaner.GetOutputPort())
    tri.Update()

    return pyvista.wrap(tri.GetOutput())


def compute_intersection(a: pyvista.PolyData, b: pyvista.PolyData) -> pyvista.PolyData:
    """
    Compute the intersection curves of two surfaces by contouring the
    signed distance from surface b evaluated on a.

    Parameters
    ----------
    a, b : pyvista.PolyData
        Surfaces.

    Returns
    -------
    pyvista.PolyData
        PolyData with line cells. Contains multiple polylines if there
        are multiple disconnected intersection curves.
    """
    if a.n_points == 0 or b.n_points == 0:
        return pyvista.PolyData()

    a_tri = convert_to_triangle_mesh(a)
    b_tri = convert_to_triangle_mesh(b)
    if a_tri.n_points == 0 or b_tri.n_points == 0:
        return pyvista.PolyData()

    # Signed distance from B:
    ippd = vtk.vtkImplicitPolyDataDistance()
    ippd.SetInput(b_tri)

    # Evaluate that distance on A:
    points = a_tri.points
    d = numpy.empty(a_tri.n_points, dtype=numpy.float32)
    for i in range(a_tri.n_points):
        x, y, z = points[i]
        d[i] = ippd.EvaluateFunction((float(x), float(y), float(z)))

    a_tri: pyvista.PolyData = a_tri.copy(deep=False)
    a_tri.point_data["__dist__"] = d

    # Contouring A at distance 0 from B yields intersection curves (as polylines on A):
    curves: pyvista.PolyData = a_tri.contour(isosurfaces=[0], scalars="__dist__")
    if curves.n_lines == 0:
        return pyvista.PolyData()

    # Merge close points:
    curves.clean(point_merging=True, tolerance=1e-9, inplace=True)
    # Larger merging tolerances drastically increase computation time.

    # Strip to maximal polylines:
    stripper = vtk.vtkStripper()
    stripper.SetInputData(curves)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    return pyvista.wrap(stripper.GetOutput())


class UnionFind:
    def __init__(self, size):
        # Initialize the parent array with each element as its own representative:
        self.parents = list(range(size))

    def find(self, i):
        # If i itself is a root:
        if self.parents[i] == i:
            return i
        # Else recursively find the representative of the parent:
        return self.find(self.parents[i])

    def unite(self, i, j):
        # Root of set containing i:
        i_root = self.find(i)
        # Root of set containing j:
        j_root = self.find(j)
        # Make the root of i's set be the root of j's set:
        self.parents[i_root] = j_root


def merge_polylines(polylines: list[NDArray], tolerance: float) -> list[NDArray]:
    """Merge polylines if their endpoints are close.

    Parameters
    ----------
    polylines : list of numpy.ndarray
        List of polylines as arrays of shape (number of points, 3).
    tolerance : float
        Distance between the endpoints of two polylines up to which the
        respective polylines will be merged.

    Returns
    -------
    list of numpy.ndarray
        List of merged polylines (including those that have not been
        merged with others) as arrays of shape (number of points, 3).
    """

    # TODO: Shorten oamc.vtk_utils.merge_polylines

    endpoint_coords = numpy.zeros((2 * len(polylines), 3), dtype=float)
    for endpoint, polyline in enumerate(polylines):
        endpoint_coords[2 * endpoint] = polyline[0]
        endpoint_coords[2 * endpoint + 1] = polyline[-1]
    tree = KDTree(endpoint_coords)
    pairs: set[tuple[int, int]] = tree.query_pairs(r=tolerance)
    uf = UnionFind(len(endpoint_coords))
    for endpoint, j in pairs:
        uf.unite(endpoint, j)
    roots = numpy.array([uf.find(i) for i in range(len(endpoint_coords))])
    vertices, vertex_from_endpoint = numpy.unique(roots, return_inverse=True)

    # Each pair of close endpoints will later be replaced by its centroid:
    vertex_coords = numpy.zeros((len(vertices), 3))
    endpoint_counts = numpy.zeros(len(vertices), int)
    for endpoint, vertex in enumerate(vertex_from_endpoint):
        vertex_coords[vertex] += endpoint_coords[endpoint]
        endpoint_counts[vertex] += 1
    vertex_coords /= endpoint_counts[:, None]

    edges: list[tuple[int, int, int]] = []  # list of (start_vertex, end_vertex, polyline_index)
    adjacency = defaultdict(list)  # map from start/end vertex to (end/start vertex, polyline)
    for polyline in range(len(polylines)):
        start_vertex = vertex_from_endpoint[2 * polyline]
        end_vertex = vertex_from_endpoint[2 * polyline + 1]
        edges.append((start_vertex, end_vertex, polyline))

        adjacency[start_vertex].append((end_vertex, polyline))
        adjacency[end_vertex].append((start_vertex, polyline))  # undirected for walking later

    visited = numpy.zeros(len(edges), dtype=bool)

    def follow_path(start_vertex: int) -> tuple[list[int], list[int]]:
        path_vertices = [start_vertex]
        path_edges = []
        current_vertex = start_vertex
        previous_vertex = None

        while True:
            # Pick the next unused edge:
            next_steps = [
                (vertex, edge)
                for vertex, edge in adjacency[current_vertex]
                if not visited[edge] and vertex != previous_vertex
            ]
            if not next_steps:
                break
            # For simple chains, there should be at most one:
            vertex, edge = next_steps[0]
            visited[edge] = True
            path_edges.append(edge)
            path_vertices.append(vertex)
            previous_vertex, current_vertex = current_vertex, vertex

        return path_vertices, path_edges

    # Start from degree-1 nodes to get open chains:
    merged_polylines = []
    for vertex in range(len(vertices)):
        if len(adjacency[vertex]) == 1:
            for neighbor, edge in adjacency[vertex]:
                if not visited[edge]:
                    path_vertices, path_edges = follow_path(vertex)
                    merged_polylines.append((path_vertices, path_edges))

    # The remaining unvisited edges are cyclic polylines or polylines
    # whose endpoints are not within the tolerance to any other
    # endpoint:
    for edge in numpy.where(visited == 0)[0]:
        start_vertex, end_vertex, polyline = edges[edge]
        # visited[edge] = True
        path_vertices, path_edges = follow_path(start_vertex)
        merged_polylines.append((path_vertices, path_edges))
        # merged_polylines.append((path_vertices, [edge] + path_edges))

    def polyline_from_edge(edge: int, start_vertex: int, end_vertex: int) -> NDArray:
        _, _, polyline = edges[edge]
        polyline = polylines[polyline].copy()
        if numpy.linalg.norm(polyline[0] - vertex_coords[start_vertex]) > numpy.linalg.norm(
            polyline[0] - vertex_coords[end_vertex]
        ):
            polyline = numpy.flip(polyline, axis=0)
        polyline[0] = vertex_coords[start_vertex]
        polyline[-1] = vertex_coords[end_vertex]
        return polyline

    merged_polyline_coords = []
    for path_vertices, path_edges in merged_polylines:
        if not path_edges:
            continue
        full_path = []
        for i, edge in enumerate(path_edges):
            polyline = polyline_from_edge(edge, path_vertices[i], path_vertices[i + 1])
            if i == 0:
                full_path.append(polyline)
            else:
                full_path.append(polyline[1:])
        merged_polyline_coords.append(numpy.vstack(full_path))

    return merged_polyline_coords


def get_polylines_from_polydata(
    lines: pyvista.PolyData,
    merge_points_tol: float = 0,
    merge_polylines_tol: float = 0,
) -> list[NDArray]:
    """Return all lines contained in a pyvista.PolyData object as arrays
    of shape (number of points, 3).

    Optionally merge close points and adjacent line segments.

    Parameters
    ----------
    lines : pyvista.PolyData
        PolyData object containing with line cells (after vtkStripper).
    merge_points_tol : float, default: 0
        Tolerance for merging close points on each polyline.
    merge_polylines_tol : float, default: 0
        Tolerance for merging adjacent polylines.

    Returns
    -------
    list of numpy.ndarray
        Lines as arrays of shape (number of points in polyline, 3).
    """

    if lines.n_lines == 0:
        return []

    nodes = lines.points

    # Connectivity array in format [node count, node 0, node 1, ..., node count, ...]:
    connectivity = lines.lines

    arrays = []
    i = 0
    while i < connectivity.size:
        n = int(connectivity[i])
        i += 1
        indices = connectivity[i : i + n]
        i += n
        array = nodes[indices]

        if merge_points_tol > 0 and array.shape[0] >= 2:
            # Merge close points:
            mask = numpy.ones(array.shape[0], dtype=bool)
            mask[1:] = numpy.linalg.norm(array[1:] - array[:-1], axis=1) > merge_points_tol
            array = array[mask]

        if array.shape[0] >= 2:
            arrays.append(array)

    if merge_polylines_tol > 0:
        arrays = merge_polylines(arrays, merge_polylines_tol)

    return arrays


def compute_int_levels(values: NDArray, offset: float = 0.0) -> NDArray:
    """
    Parameters
    ----------
    values : numpy.ndarray
        Values between which all integer levels shall be computed.
    offset : float, default: 0.0
        Offset from integer levels.

    Returns
    -------
    numpy.ndarray
        All integers in the interval [min(values), max(values)].

    Examples
    --------
    >>> compute_int_levels(values=numpy.array([0.8, 1.4, 0.3, 0.6, 3.1]), offset=0.6)
    array([0.6, 1.6, 2.6])
    """
    min_level = int(numpy.ceil(numpy.min(values) - offset))
    max_level = int(numpy.floor(numpy.max(values) - offset))

    if min_level > max_level:
        return numpy.array(
            object=[],
            dtype=values.dtype,
        )

    return (
        numpy.arange(
            start=min_level,
            stop=max_level + 1,
            dtype=values.dtype,
        )
        + offset
    )


def compute_int_isosurface_intersections(
    grid: pyvista.UnstructuredGrid,
    p: NDArray,
    q: NDArray,
    *,
    p_splits: int = 1,
    p_name: str = "p",
    q_name: str = "q",
    clean_mesh: bool = False,
) -> dict[tuple[int, int], list[numpy.ndarray]]:
    """
    Compute integer-level isosurface intersection curves of the given
    scalar fields.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The grid.
    p, q : numpy.ndarray
        1D arrays of nodal values.
    p_splits : int, default: 1
        TODO: Explain p_splits.
    p_name, q_name : str, default: "p", "q"
        Names to attach the arrays to the mesh for contouring.
    clean_mesh : bool, default: False
        If True, run grid.clean() (may help if the mesh has duplicated points).

    Returns
    -------
    dict of tuple of int to list of numpy.ndarray
        Map from integer level pair (i, j,) to a list of polylines.
        Each polyline is an array of shape (n_points, 3,).
    """

    if p.shape[0] != grid.n_points or q.shape[0] != grid.n_points:
        raise ValueError("p and q must have length grid.n_points.")

    # Work on a shallow copy to avoid mutating the dataset of the caller:
    grid_copy: pyvista.UnstructuredGrid = grid.copy(deep=False)
    grid_copy.point_data[p_name] = p
    grid_copy.point_data[q_name] = q

    if clean_mesh:
        # Clean coincident points/cells:
        grid_copy = grid_copy.clean(tolerance=0.0)
        # May slightly reorder point IDs.

    p_offsets = numpy.arange(p_splits) / p_splits

    p_levels = numpy.concatenate([compute_int_levels(p, offset=offset) for offset in p_offsets])
    p_levels.sort()

    q_levels = compute_int_levels(q)

    if p_levels.size == 0 or q_levels.size == 0:
        return dict()

    # Precompute isosurfaces and bounds to skip empty/non-overlapping pairs:
    p_surfaces: dict[float, pyvista.PolyData] = {}
    q_surfaces: dict[float, pyvista.PolyData] = {}
    p_surface_bounds: dict[float, tuple[float, float, float, float, float, float]] = {}
    q_surface_bounds: dict[float, tuple[float, float, float, float, float, float]] = {}

    for p_level in p_levels:
        surface = compute_level_surface(grid_copy, p_name, p_level)
        p_surfaces[p_level] = surface
        p_surface_bounds[p_level] = (
            surface.bounds
            if surface.n_points
            else (numpy.inf, -numpy.inf, numpy.inf, -numpy.inf, numpy.inf, -numpy.inf)
        )

    for q_level in q_levels:
        surface = compute_level_surface(grid_copy, q_name, q_level)
        q_surfaces[q_level] = surface
        q_surface_bounds[q_level] = (
            surface.bounds
            if surface.n_points
            else (numpy.inf, -numpy.inf, numpy.inf, -numpy.inf, numpy.inf, -numpy.inf)
        )

    def _bounds_overlap(a, b) -> bool:
        return not (
            a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2] or a[5] < b[4] or b[5] < a[4]
        )

    intersections_all_levels: dict[tuple[int, int], list[NDArray]] = {}

    for q_level in q_levels:
        q_surface = q_surfaces[q_level]
        if q_surface.n_points == 0:
            continue

        for p_level in p_levels[numpy.arange(int(q_level % p_splits), len(p_levels), p_splits)]:
            p_surface = p_surfaces[p_level]
            if p_surface.n_points == 0:
                continue

            if not _bounds_overlap(p_surface_bounds[p_level], q_surface_bounds[q_level]):
                continue

            intersections = compute_intersection(p_surface, q_surface)
            if intersections.n_lines == 0:
                continue

            intersections = get_polylines_from_polydata(
                lines=intersections,
                merge_points_tol=1e-3,
                merge_polylines_tol=1e-2,
            )
            if intersections:
                intersections_all_levels[(p_level, q_level)] = intersections

    return intersections_all_levels


def rectangular_tube(
    p: NDArray,
    z: NDArray,
    w: float,
    h: float,
    cap: bool = True,
    scalars: NDArray | None = None,
) -> pyvista.PolyData:
    """Build a rectangular tube mesh along a path where each
    cross-section is oriented along a local z-axis.

    Parameters
    ----------
    p : numpy.ndarray
        Points defining the path.
    z : numpy.ndarray
        Vectors aligned with local z-axes, not necessarily unit vectors.
    w : float
        Size of the rectangular cross section in local x-direction.
    h : float
        Size of the rectangular cross section in local y-direction.
    cap : bool, default: True
        Whether to add close the ends of the tube.
    scalars : numpy.ndarray, optional
        Scalars for coloring the path, for example.

    Returns
    -------
    pyvista.PolyData
        Path as a 3D mesh.
    """

    p = numpy.asarray(p, dtype=float)
    z = numpy.asarray(z, dtype=float)
    if p.shape != z.shape:
        raise ValueError("Points and z-axes must have the same shape.")

    # Number of points defining the path:
    n = len(p)
    if n < 2:
        raise ValueError("Need at least 2 points for a path")

    # Determine tangent directions:
    t = numpy.zeros_like(p)
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    # Tangent and point (i) = direction of vector from point (i - 1) to
    # point (i + 1) for i in (1, n):
    t[1:-1] = p[2:] - p[:-2]
    t /= numpy.linalg.norm(t, axis=1)[:, None]

    z /= numpy.linalg.norm(z, axis=1)[:, None]

    # Half dimensions:
    hw = w / 2
    hh = h / 2

    # Create vertices:
    vertices = []
    for pi, ti, zi in zip(p, t, z):
        zi = zi - numpy.dot(zi, ti) * ti
        nzi = numpy.linalg.norm(zi)
        if nzi < 1e-9:
            raise ValueError(
                "The local z-axis must not be aligned with the local tangent direction."
            )
        zi /= nzi

        yi = numpy.cross(zi, ti)
        yi /= numpy.linalg.norm(yi)

        corners = [
            pi + hw * yi - hh * zi,
            pi - hw * yi - hh * zi,
            pi - hw * yi + hh * zi,
            pi + hw * yi + hh * zi,
        ]
        vertices.extend(corners)

    vertices = numpy.array(vertices)

    # Create faces:
    v = len(vertices)
    faces = [4, 0, 1, 2, 3, 4, v - 4, v - 3, v - 2, v - 1] if cap else []
    for i in range(n - 1):
        a = 4 * i  # first vertex of cross section a
        b = 4 * (i + 1)  # first vertex of cross section b
        quads = [
            [4, a + 0, a + 1, b + 1, b + 0],
            [4, a + 1, a + 2, b + 2, b + 1],
            [4, a + 2, a + 3, b + 3, b + 2],
            [4, a + 3, a + 0, b + 0, b + 3],
        ]
        for q in quads:
            faces.extend(q)

    faces = numpy.array(faces, dtype=int)

    # Create mesh:
    mesh = pyvista.PolyData(vertices, faces)

    if scalars is not None:
        scalars = numpy.asarray(scalars)
        if len(scalars) != n:
            raise ValueError("There must be one scalar per point.")
        mesh["scalars"] = numpy.repeat(scalars, 4)

    return mesh
