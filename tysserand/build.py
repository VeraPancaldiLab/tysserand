"""
Network reconstruction from node coordinates.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix

from .convert import _coalesce, to_undirected, vectors_to_csr
from .distances import distance_neighbors, find_trim_dist

__all__ = [
    'build_delaunay',
    'build_knn',
    'build_rdn',
    'build_lattice',
    'link_solitaries',
]


def build_delaunay(
    coords: np.ndarray,
    min_dist: float = 0,
    trim_dist: str | float | None = 'percentile_size',
    perc: float = 99,
    node_adaptive_trimming: bool = False,
    n_edges: int = 3,
    trim_dist_ratio: float = 2,
    return_dist: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    """
    Reconstruct edges between nodes by Delaunay triangulation, returning a CSR adjacency matrix.
    Use `trim_dist` to remove edges above a threshold distance.
    Use `node_adaptive_trimming` to remove edges above a threshold
    distance adapted to the number of neighbors of each node.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...).
    min_dist : float, optional
        Minimum distance threshold used with the node adaptive trimming method.
    trim_dist : str, float, or False, optional
        Method or distance used to delete reconstructed edges.
        Use 'percentile_size' to adapt this distance to the number of nodes.
        Set to False to skip global trimming.
    perc : float, optional
        Percentile of distances used as the threshold when method is 'percentile'.
        The default is 99.
    node_adaptive_trimming : bool, optional
        Whether to trim edges with a distance threshold adapted for each node.
        For each node, the threshold is the Nth smallest edge length x `trim_dist_ratio`.
        Edges above this threshold and above `min_dist` are discarded.
    n_edges : int, optional
        The Nth smallest edge used to compute the per-node trimming distance.
    trim_dist_ratio : float, optional
        Ratio between the distance threshold and the Nth smallest distance.
    return_dist : bool, optional
        Whether to also return the CSR distance matrix. The default is False.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix of the reconstructed network.
    D : scipy.sparse.csr_matrix
        CSR matrix of edge distances, only returned when `return_dist` is True.

    Examples
    --------
    >>> coords = make_simple_coords()
    >>> A = build_delaunay(coords, trim_dist=False)
    """
    pairs = Voronoi(coords).ridge_points
    A = vectors_to_csr(pairs)

    if node_adaptive_trimming or trim_dist is not False or return_dist:
        D = distance_neighbors(coords, A)

    if node_adaptive_trimming:
        trim_dist = False
        for node_id in range(len(coords)):
            d_row = D.getrow(node_id)
            neighbor_ids = d_row.indices
            node_distances = d_row.data
            if len(node_distances) > n_edges:
                thresh_dist = np.sort(node_distances)[n_edges - 1] * trim_dist_ratio
                discard_mask = (node_distances > thresh_dist) & (node_distances > min_dist)
                for j in neighbor_ids[discard_mask]:
                    D[node_id, j] = 0
                    D[j, node_id] = 0
        D.eliminate_zeros()
        A = (D > 0).astype(np.int8)

    if trim_dist is not False:
        if not isinstance(trim_dist, (int, float)):
            trim_dist = find_trim_dist(D, method=trim_dist, perc=perc)
        D.data[D.data >= trim_dist] = 0
        D.eliminate_zeros()
        A = D.astype(bool).astype(np.int8)

    if return_dist:
        return A, D
    return A


def build_knn(
    coords: np.ndarray,
    k: int = 6,
    approximate: bool = True,
    metric: str = 'euclidean',
    n_jobs: int = -1,
    undirected: bool = True,
    random_state: int | np.random.RandomState | None = None,
    **kwargs,
) -> csr_matrix:
    """
    Reconstruct edges between nodes by k-nearest neighbors (knn) method,
    returning a CSR adjacency matrix.
    An edge is drawn between each node and its k nearest neighbors.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...).
    k : int, optional
        Number of nearest neighbors. The default is 6.
    approximate : bool, optional
        Whether fast approximation is authorized for networks >= 3000 nodes.
    metric : str or callable, optional
        The metric to use for computing nearest neighbors. The default is 'euclidean'.
        See pynndescent documentation.
    n_jobs : int or None, optional
        The number of parallel jobs to run for neighbors index construction.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    random_state : None, int or numpy.random.RandomState, optional
        Seed of the approximate neighbors search (pynndescent, networks >= 3000
        nodes). An int gives reproducible networks, but disables some of
        pynndescent's parallelism. The default is None.

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR adjacency matrix of the reconstructed network.

    Examples
    --------
    >>> coords = make_simple_coords()
    >>> A = build_knn(coords)
    """
    from sklearn.neighbors import BallTree
    import pynndescent
    if len(coords) < 3000 or not approximate:
        tree = BallTree(coords, **kwargs)
        _, neighbors = tree.query(coords, k=k + 1)  # first neighbor is oneself
    else:
        n_index = min(k, len(coords))
        # compiles pynndescent's numba functions on a tiny dataset
        warmup_coords = np.random.default_rng(0).random(size=(n_index + 2, coords.shape[1]))
        warmup_index = pynndescent.NNDescent(
            warmup_coords,
            metric=metric,
            n_neighbors=n_index + 1,
            n_jobs=n_jobs,
        )
        index = pynndescent.NNDescent(
            coords,
            metric=metric,
            n_neighbors=n_index + 1,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        neighbors = index.neighbor_graph[0]
    
    # convert from BallTree query format to CSR matrix
    n_nodes = neighbors.shape[0]
    src = np.repeat(neighbors[:, 0], k)
    tgt = neighbors[:, 1:].ravel()
    data = np.ones(len(src), dtype=np.int8)
    A = csr_matrix((data, (src, tgt)), shape=(n_nodes, n_nodes))
    A = A.maximum(A.T).astype(np.int8)
    if undirected:
        A = A.maximum(A.T)

    return A.astype(np.int8)


def build_rdn(
    coords: np.ndarray,
    r: float,
    coords_ref: np.ndarray = None,
    **kwargs,
) -> csr_matrix:
    """
    Reconstruct edges between nodes by radial distance neighbors (rdn) method,
    returning a CSR adjacency matrix.
    An edge is drawn between each node and the nodes closer
    than a threshold distance (within a radius).

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...).
    r : float
        Radius within which nodes are connected.
    coords_ref : np.ndarray, optional
        Source points in the network. If provided, edges go from `coords_ref` to `coords`
        and the output matrix has shape (len(coords_ref), len(coords)).
        If None, `coords` is used and the network is undirected.

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR adjacency matrix of the reconstructed network.
        Symmetric of shape (n, n) when `coords_ref` is None,
        or directed of shape (n_ref, n) otherwise.

    Examples
    --------
    >>> coords = make_simple_coords()
    >>> A = build_rdn(coords, r=60)
    """
    from sklearn.neighbors import BallTree
    tree = BallTree(coords, **kwargs)
    if coords_ref is None:
        query_coords = coords
        n_src = n_tgt = len(coords)
    else:
        query_coords = coords_ref
        n_src, n_tgt = len(coords_ref), len(coords)
    ind = tree.query_radius(query_coords, r=r)
    # Find the self-index of each query point in the tree (distance == 0 means the
    # query point is present in the tree and would create a self-loop).
    self_dists, self_idx = tree.query(query_coords, k=1)
    self_dists = self_dists.ravel()
    self_idx = self_idx.ravel()
    src_list = []
    tgt_list = []
    for i, arr in enumerate(ind):
        neigh = arr[arr != self_idx[i]] if self_dists[i] == 0 else arr
        src_list.append(np.full(len(neigh), i, dtype=int))
        tgt_list.append(neigh)
    src = np.concatenate(src_list)
    tgt = np.concatenate(tgt_list)
    data = np.ones(len(src), dtype=np.int8)
    A = csr_matrix((data, (src, tgt)), shape=(n_src, n_tgt))
    if coords_ref is None:
        A = A.maximum(A.T).astype(np.int8)
    return A


def build_lattice(
    coords: np.ndarray,
    r: float = None,
    coords_ref: np.ndarray = None,
    lattice: str = 'hexagonal',
    r_factor: float = None,
    **kwargs,
) -> csr_matrix:
    """
    Reconstruct edges between nodes organized on a lattice, returning a CSR adjacency matrix.
    The lattice can be hexagonal, square, or cubic (3D).
    An edge is drawn between each node and the nodes closer
    than a threshold distance (within a radius).

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...).
    r : float, optional
        Radius within which nodes are connected. If not provided, it is guessed from data.
    coords_ref : np.ndarray, optional
        Source points in the network. If provided, edges go from `coords_ref` to `coords`
        and the output matrix has shape (len(coords_ref), len(coords)).
        If None, `coords` is used and the network is undirected.
    lattice : str, optional
        Type of spatial organization of nodes. Can be 'hexagonal', 'square', or 'cubic'.
        Used to set `r_factor` if not provided. The default is 'hexagonal'.
    r_factor : float, optional
        Multiplicative factor applied to the minimum inter-node distance to obtain `r`.
        Inferred from `lattice` if not provided.

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR adjacency matrix of the reconstructed network.
        Symmetric of shape (n, n) when `coords_ref` is None,
        or directed of shape (n_ref, n) otherwise.

    Examples
    --------
    >>> coords = make_simple_coords()
    >>> A = build_lattice(coords)
    """
    from sklearn.neighbors import BallTree
    tree = BallTree(coords, **kwargs)
    if r is None:
        # derive minimum inter-node distance from nearest neighbor distances
        min_dists, _ = tree.query(coords, k=2)  # column 0 is self (dist=0)
        min_dist = min_dists[:, 1].min()
        if r_factor is None:
            if lattice == 'hexagonal':
                r_factor = 1.5
            elif lattice == 'square':
                r_factor = np.sqrt(2) * 1.01
            elif lattice == 'cubic':
                r_factor = np.sqrt(3) * 1.01
        r = min_dist * r_factor
    return build_rdn(coords, r=r, coords_ref=coords_ref, **kwargs)


def link_solitaries(
    coords: np.ndarray,
    A: csr_matrix,
    method: str = 'delaunay',
    min_neighbors: int = 3,
    all_pairs: np.ndarray | None = None,
    all_dist: np.ndarray | None = None,
    verbose: int = 1) -> csr_matrix:
    """
    Detect nodes that are not connected (or under-connected) and link them to
    other nodes, returning an updated CSR adjacency matrix.

    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...).
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix of the network, possibly directed or weighted.
        Node degrees ignore edge directions (i and j are neighbors if
        ``A[i, j]`` or ``A[j, i]`` is non-zero).
    method : str, optional
        Method used to connect solitary nodes to their neighbors. Default is
        'delaunay'. With 'knn', solitary nodes are connected to their nearest
        neighbors.
    min_neighbors : int, optional
        Minimum number of neighbors before linking nodes to additional nodes.
    all_pairs : scipy.sparse.csr_matrix, optional
        Pre-computed adjacency matrix of potential edges to re-link nodes.
    all_dist : scipy.sparse.csr_matrix, optional
        Pre-computed CSR matrix of distances between potential pairs of nodes.
    verbose : int, optional
        Verbosity, if different from 0 some messages are displayed. Default is 1.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Updated CSR adjacency matrix, with additional edges for solitary nodes.
        Existing edges keep their values and directions; new edges are
        undirected (stored in both directions) with a value of 1.

    Example
    -------
    >>> coords = np.array([[0, 0], [1, 0], [2, 0], [3.1, 0], [4, 0]])
    >>> A = vectors_to_csr(np.array([[0, 1], [1, 2]]), n_nodes=5)
    >>> A = link_solitaries(coords, A, method='knn', min_neighbors=1)
    """
    n_nodes = coords.shape[0]
    A = A.tocsr(copy=True)
    if A.shape != (n_nodes, n_nodes):
        # nodes without edges at the end may be missing from A
        A.resize((n_nodes, n_nodes))
    A = _coalesce(A)
    # neighbors regardless of edge directions and values
    A_und = to_undirected(A)

    # detect nodes with no edges, or too few edges (self-loops excluded)
    degrees = np.diff(A_und.indptr) - A_und.diagonal()
    solitaries = np.flatnonzero(degrees < min_neighbors).tolist()
    n_solitaries = len(solitaries)

    if n_solitaries == 0:
        if verbose != 0:
            if min_neighbors == 1:
                print("all nodes have at least one edge")
            else:
                print(f"all nodes have at least {min_neighbors} edges")
    else:
        if verbose != 0:
            if min_neighbors == 1:
                print(f"there are {n_solitaries}/{n_nodes} nodes with no edges")
            else:
                print(f"there are {n_solitaries}/{n_nodes} nodes with < {min_neighbors} edges")

        new_pairs = []
        if method == 'delaunay':
            if all_pairs is None:
                all_pairs = build_delaunay(coords, node_adaptive_trimming=False, trim_dist=False)
            if all_dist is None:
                all_dist = distance_neighbors(coords, all_pairs)
            for node_id in solitaries:
                cand = all_pairs.indices[all_pairs.indptr[node_id]:all_pairs.indptr[node_id + 1]]
                node_distances = all_dist.data[all_dist.indptr[node_id]:all_dist.indptr[node_id + 1]]
                if len(node_distances) >= min_neighbors:
                    keep = np.argsort(node_distances)[:min_neighbors]
                    for j in cand[keep]:
                        new_pairs.append((node_id, int(j)))
        elif method == 'knn':
            nn_A = build_knn(coords, k=min_neighbors)
            for i in solitaries:
                for j in nn_A.indices[nn_A.indptr[i]:nn_A.indptr[i + 1]]:
                    new_pairs.append((i, int(j)))

        if len(new_pairs) > 0:
            A_new = to_undirected(vectors_to_csr(np.array(new_pairs), n_nodes=n_nodes))
            # only add edges absent from A, so existing values are kept
            A_new = A_new - A_new.multiply(A.astype(bool))
            A_new.eliminate_zeros()
            A = (A + A_new.astype(A.dtype)).tocsr()

    return A
