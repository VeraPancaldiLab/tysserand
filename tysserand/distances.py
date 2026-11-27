"""
Edge distances, distance-based trimming and weights.
"""
import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    'distance_neighbors',
    'find_trim_dist',
    'weight_from_distance',
]


def distance_neighbors(
    coords: np.ndarray,
    A: csr_matrix,
) -> csr_matrix:
    """
    Compute edge lengths from a CSR adjacency matrix, returning a CSR matrix of distances.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of points, shape (n_nodes, n_dims).
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix of shape (n_nodes, n_nodes).

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR matrix of the same sparsity structure as `A`, where each non-zero entry
        holds the Euclidean distance between the corresponding pair of nodes.
    """
    rows, cols = A.nonzero()
    distances = np.sqrt(((coords[rows] - coords[cols]) ** 2).sum(axis=1))
    return csr_matrix((distances, (rows, cols)), shape=A.shape)


def find_trim_dist(
    D: csr_matrix,
    method: str = 'percentile_size',
    perc: float = 99,
) -> float:
    """
    Find the distance threshold to eliminate reconstructed edges in a network.

    Parameters
    ----------
    D : scipy.sparse.csr_matrix
        CSR matrix of edge distances as returned by `distance_neighbors`.
    method : str, optional
        Method used to compute the threshold. The default is 'percentile_size',
        which derives an optimal percentile from the number of nodes.
    perc : float, optional
        Percentile of distances used as the threshold when method is 'percentile'.
        The default is 99.

    Returns
    -------
    float
        Threshold distance.
    """
    dist = D.data
    if method == 'percentile_size':
        nb_nodes = D.shape[0]
        prop_edges = 4 / nb_nodes ** 0.5
        perc = 100 * (1 - prop_edges * 0.5)
    return float(np.percentile(dist, perc))


def weight_from_distance(
    D: csr_matrix,
    weight_range: tuple = None,
) -> csr_matrix:
    """
    Compute edge weights from a CSR distance matrix, with shorter distances
    receiving higher weights (inverse linear mapping).

    Parameters
    ----------
    D : scipy.sparse.csr_matrix
        CSR matrix of edge distances as returned by `distance_neighbors`.
    weight_range : tuple of (float, float), optional
        Custom weight range (low, high). The shortest distance is mapped to `high`
        and the longest distance to `low`. If None, the range is automatically set
        to (d_min / d_max, 1) where d_min and d_max are the minimum and maximum
        distances. The default is None.

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR matrix with the same sparsity structure as `D`, where each non-zero
        entry holds the computed weight.

    Examples
    --------
    >>> coords = make_simple_coords()
    >>> A = build_delaunay(coords, trim_dist=False)
    >>> D = distance_neighbors(coords, A)
    >>> W = weight_from_distance(D)
    >>> W_custom = weight_from_distance(D, weight_range=(0.1, 5.0))
    """
    W = D.copy()

    if len(W.data) == 0:
        return W

    d_min = W.data.min()
    d_max = W.data.max()

    if weight_range is None:
        w_low = d_min / d_max
        w_high = 1.0
    else:
        w_low, w_high = weight_range

    if d_max == d_min:
        W.data[:] = w_high
    else:
        W.data = w_high - (W.data - d_min) * (w_high - w_low) / (d_max - d_min)

    return W
