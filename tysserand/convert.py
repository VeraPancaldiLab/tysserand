"""
Conversions between network representations (CSR adjacency, edge pairs,
DataFrames) and attribute helpers.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, triu

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


def _warn_legacy_pairs(fct_name):
    warnings.warn(
        f"`{fct_name}` with edge-list arrays is deprecated, pass CSR adjacency "
        "matrices instead (see `vectors_to_csr` to convert edge pairs).",
        DeprecationWarning, stacklevel=3)


def remove_duplicate_pairs(A, out_dtype=np.int8):
    """
    Remove duplicate edges and edge directions from a network.

    Each undirected edge ends up stored once per direction, with a value of 1,
    whatever its multiplicity or direction(s) in the input.

    Parameters
    ----------
    A : scipy.sparse matrix
        Adjacency matrix, possibly directed, with duplicate (summed or
        non-coalesced) entries. Passing a (n_pairs x 2) array of node pairs
        is deprecated: it returns the sorted unique pairs as before.
    out_dtype : np.dtype, optional
        Data type of the output matrix, by default np.int8.

    Returns
    -------
    A_uniq : scipy.sparse.csr_matrix
        Symmetric binary CSR adjacency matrix.

    Example
    -------
    >>> A = vectors_to_csr_directed(np.array([[4, 3], [1, 2], [3, 4], [2, 1]]))
    >>> csr_to_edge_pairs(remove_duplicate_pairs(A))
    array([[1, 2],
           [3, 4]])
    """
    if not issparse(A):
        _warn_legacy_pairs('remove_duplicate_pairs')
        return np.unique(np.sort(A, axis=1), axis=0)
    # boolean sums are logical OR: no overflow nor cancellation of entries
    A = csr_matrix(A).astype(bool)
    A.eliminate_zeros()
    A = (A + A.T).tocsr()
    A.sum_duplicates()
    return A.astype(out_dtype)


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


def pairs_to_df(A, columns=['source', 'target'], weight=None):
    """
    Convert a network into a dataframe of edges, one row per undirected edge.

    Parameters
    ----------
    A : scipy.sparse matrix
        Symmetric adjacency matrix. Passing a (n_pairs x 2) array of node
        pairs is deprecated: its rows are used as they are.
    columns : Index or array-like
        Column labels of the edges nodes. Default is ['source', 'target']
    weight : str, optional
        If given, name of an additional column holding the values of ``A``
        (edge weights, distances...).

    Returns
    -------
    edges : dataframe
        Edges indicated by the nodes 'source' and 'target' they link,
        with source < target.

    Example
    -------
    >>> A = vectors_to_csr(np.array([[1, 0], [1, 2]]))
    >>> pairs_to_df(A)
       source  target
    0       0       1
    1       1       2
    """
    if not issparse(A):
        _warn_legacy_pairs('pairs_to_df')
        return pd.DataFrame(data=A, columns=columns)
    A = triu(A, k=1, format='coo')
    A.sum_duplicates()  # canonical row-major order, coalesced entries
    A.eliminate_zeros()
    edges = pd.DataFrame({columns[0]: A.row, columns[1]: A.col})
    if weight is not None:
        edges[weight] = A.data
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
