"""
Conversions between network representations (CSR adjacency, edge pairs,
DataFrames) and attribute helpers.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

__all__ = [
    'remove_duplicate_pairs',
    'vectors_to_csr',
    'vectors_to_csr_directed',
    'csr_to_edge_pairs',
    'csr_to_edge_coords',
    'attributes_to_matrix',
    'categorical_to_integer',
    'flatten_categories',
    'coords_to_df',
    'pairs_to_df',
    'double_sort',
]


def remove_duplicate_pairs(pairs):
    """
    Remove redundant rows in a 2D array.
    
    Parameters
    ----------
    pairs : ndarray
        The (n_pairs x 2) array of neighbors indices.

    Returns
    -------
    uniq_pairs : ndarray
        Array of unique pairs, the content of each row is sorted.
    
    Example
    -------
    >>> pairs = [[4, 3],
                 [1, 2],
                 [3, 4],
                 [2, 1]]
    >>> remove_duplicate_pairs(pairs)
    array([[1, 2],
           [3, 4]])
    """
    
    uniq_pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return uniq_pairs


def vectors_to_csr(
    pairs: np.ndarray,
    n_nodes: int = None,
    out_dtype: np.dtype = np.int8,
) -> csr_matrix:
    """
    Convert an edge list to an undirected CSR adjacency matrix.

    Parameters
    ----------
    pairs : np.ndarray
        Array of shape (N_edges, 2) where each row is a [source, target] node index pair.
    n_nodes : int, optional
        Number of nodes. If None, inferred as ``pairs.max() + 1``. Pass explicitly
        when isolated nodes (with no edges) are present to avoid a too-small matrix.
    out_dtype : np.dtype, optional
        Data type of the output matrix, by default np.int8.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Symmetric CSR adjacency matrix of shape (n_nodes, n_nodes) with entries of `out_dtype`.
    """
    if n_nodes is None:
        n_nodes = int(pairs.max()) + 1
    data = np.ones(len(pairs) * 2)  # undirected: store both directions
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    A = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).astype(out_dtype)
    return A


def vectors_to_csr_directed(
    pairs: np.ndarray,
    n_nodes: int = None,
    out_dtype: np.dtype = np.int8,
) -> csr_matrix:
    """
    Convert an edge list to a directed CSR adjacency matrix.

    Parameters
    ----------
    pairs : np.ndarray
        Array of shape (N_edges, 2) where each row is a [source, target] node index pair.
    n_nodes : int, optional
        Number of nodes. If None, inferred as ``pairs.max() + 1``. Pass explicitly
        when isolated nodes (with no edges) are present to avoid a too-small matrix.
    out_dtype : np.dtype, optional
        Data type of the output matrix, by default np.int8.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Asymmetric CSR adjacency matrix of shape (n_nodes, n_nodes) with entries of `out_dtype`,
        where A[i, j] = 1 indicates an edge from node i to node j.
    """
    if n_nodes is None:
        n_nodes = int(pairs.max()) + 1
    data = np.ones(len(pairs))
    rows = pairs[:, 0]
    cols = pairs[:, 1]
    A = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).astype(out_dtype)
    return A


def csr_to_edge_pairs(A: csr_matrix) -> np.ndarray:
    """
    Extract upper-triangle edge pairs from an undirected CSR adjacency matrix.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Symmetric CSR adjacency matrix.

    Returns
    -------
    pairs : np.ndarray
        Array of shape (n_edges, 2) with one row per undirected edge.
    """
    rows, cols = A.nonzero()
    mask = rows < cols
    return np.vstack([rows[mask], cols[mask]]).T


def csr_to_edge_coords(coords: np.ndarray, A: csr_matrix) -> list:
    """
    Convert a CSR adjacency matrix to a list of edge coordinate arrays
    suitable for a napari Shapes layer.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates in napari convention (row, col order), shape (n_nodes, ndim).
        These should already have been flipped by :func:`convert_nodes_tys_to_nap`.
    A : scipy.sparse.csr_matrix
        Symmetric CSR adjacency matrix.

    Returns
    -------
    edges_coords : list of np.ndarray
        List of (2, ndim) arrays, one per undirected edge, giving the coordinates
        of its two endpoints.
    """
    pairs = csr_to_edge_pairs(A)
    return [coords[pair] for pair in pairs]


def attributes_to_matrix(nodes, attributes):
    """
    Build the ``(n_nodes, n_attributes)`` matrix ``C`` of node attributes aligned
    to the adjacency matrix node order (positional index ``0..n-1``). It is used,
    for instance, for the matrix form of the mixing matrix ``M = C.T @ A @ C``.

    Parameters
    ----------
    nodes : pd.DataFrame or array-like
        Node attributes. If a DataFrame, ``attributes`` selects its columns.
    attributes : str or list
        Attribute column(s) to extract.

    Returns
    -------
    C : np.ndarray
        Float matrix of shape ``(n_nodes, n_attributes)``.
    """
    if isinstance(attributes, str):
        attributes = [attributes]
    if isinstance(nodes, pd.DataFrame):
        C = nodes.loc[:, list(attributes)].to_numpy(dtype=float)
    else:
        C = np.asarray(nodes, dtype=float)
        if C.ndim == 1:
            C = C[:, np.newaxis]
    return C


def categorical_to_integer(l):
    uniq = set(l)
    nb_uniq = len(uniq)
    mapping = dict(zip(uniq, range(nb_uniq)))
    converted = [mapping[x] for x in l]
    return converted


def flatten_categories(nodes, att):
    # the reverse operation is 
    # nodes = nodes.join(pd.get_dummies(nodes['nodes_class']))
    return nodes.loc[:, att].idxmax(axis=1)


def coords_to_df(coords, columns=None):
    """
    Convert an array of coordinates of nodes into a dataframe.

    Parameters
    ----------
    coords : ndarray
        Coordinates of points with columns corresponding to axes ('x', 'y', ...)
    columns : Index or array-like
        Column labels to use for resulting frame. Will default to
        ['x0', 'x1',..., 'xn'] if no column labels are provided.

    Returns
    -------
    nodes : dataframe
        Coordinates of nodes indicated by 'x', 'y' or other if required.
    """
    nb_dim = coords.shape[1]
    if columns is None:
        if nb_dim == 2:
            columns = ['x', 'y']
        elif nb_dim == 3:
            columns = ['x', 'y', 'z']
        else:
            columns = ['x'+str(i) for i in range(nb_dim)]
    
    nodes = pd.DataFrame(data=coords, columns=columns)
    return nodes


def pairs_to_df(pairs, columns=['source', 'target']):
    """
    Convert an array of pairs of nodes into a dataframe

    Parameters
    ----------
    pairs : ndarray
        The (n_pairs x 2) array of neighbors indices.
    columns : Index or array-like
        Column labels to use for resulting frame. Default is ['source', 'target']

    Returns
    -------
    edges : dataframe
        Edges indicated by the nodes 'source' and 'target' they link.
    """

    edges = pd.DataFrame(data=pairs, columns=columns)
    return edges


def double_sort(data, last_var=0):
    """
    Sort twice an array, first on axis 1, then preserves
    whole rows and sort by one column on axis 0.
    Usefull to compare pairs of nodes obtained
    with different methods.

    Parameters
    ----------
    data : 2D array
        Data to sort.
    last_var : int, optional. The default is 0.
        Column by which intact rows are sorted.

    Returns
    -------
    data : 2D array
        Sorted data.
        
    Examples
    --------
    >>> pairs = np.array([[4,3],
                          [5,6],
                          [2,1]])
    >>> double_sort(pairs)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """
    
    # doing simply np.sort(np.sort(pairs, axis=1), axis=0)
    # would uncouple first and second elements of pairs
    # during the second sorting (axis=0)
    data = np.sort(data, axis=1)
    x_sort = np.argsort(data[:, 0])
    data = data[x_sort]
    
    return data
