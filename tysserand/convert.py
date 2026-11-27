"""
Conversions between network representations (CSR adjacency, edge pairs,
DataFrames) and attribute helpers.
"""

from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, triu

from typing import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from numpy.typing import DTypeLike

__all__ = [
    'is_directed',
    'to_undirected',
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


def _combine_values(A, B, combine):
    """Element-wise combination of two sparse matrices, entries absent from
    one of them being kept as they are in the other one."""
    if combine == 'max':
        return A.maximum(B)
    if combine == 'sum':
        return A + B
    raise ValueError(f"`combine` must be 'max' or 'sum', not {combine!r}")


def _coalesce(A):
    """CSR copy of A with summed duplicate entries and no explicit zeros."""
    A = csr_matrix(A, copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


def is_directed(A: csr_matrix) -> bool:
    """True if the adjacency matrix ``A`` is not symmetric (edge directions
    or weights differ between ``A[i, j]`` and ``A[j, i]``)."""
    A = _coalesce(A)
    return (A != A.T).nnz > 0


def to_undirected(A: csr_matrix, weighted: bool = False, combine: str = 'max', out_dtype: DTypeLike | None = None) -> csr_matrix:
    """
    Convert a network to an undirected (symmetric) network.

    Parameters
    ----------
    A : scipy.sparse matrix
        Adjacency matrix, possibly directed, weighted or with non-coalesced
        duplicate entries (which are summed, following scipy).
    weighted : bool, optional
        If False (default), the output is binary: an edge exists between i
        and j if ``A[i, j]`` or ``A[j, i]`` is non-zero. If True, edge values
        are kept, ``A[i, j]`` and ``A[j, i]`` being combined with `combine`.
    combine : {'max', 'sum'}, optional
        How the values of the 2 directions of an edge are combined when
        ``weighted=True``. With 'max' (default) a symmetric network is left
        unchanged; with 'sum' the values of reciprocal edges are added, so a
        symmetric network has its values doubled.
    out_dtype : np.dtype, optional
        Data type of the output matrix. Default is np.int8 for binary
        outputs, the input data type for weighted outputs.

    Returns
    -------
    A_und : scipy.sparse.csr_matrix
        Symmetric CSR adjacency matrix.

    Example
    -------
    >>> A = vectors_to_csr_directed(np.array([[4, 3], [1, 2], [3, 4], [2, 1]]))
    >>> csr_to_edge_pairs(to_undirected(A))
    array([[1, 2],
           [3, 4]])
    """
    if weighted:
        A = _coalesce(A)
        A = _combine_values(A, A.T.tocsr(), combine)
        A.eliminate_zeros()
        return A if out_dtype is None else A.astype(out_dtype)
    # boolean sums are logical OR: no overflow nor cancellation of entries
    A = csr_matrix(A).astype(bool)
    A.eliminate_zeros()
    A = (A + A.T).tocsr()
    A.sum_duplicates()
    return A.astype(np.int8 if out_dtype is None else out_dtype)


def remove_duplicate_pairs(A: csr_matrix | np.ndarray, out_dtype: DTypeLike = np.int8) -> csr_matrix | np.ndarray:
    """
    Deprecated, use :func:`to_undirected`.

    Convert a network into a symmetric binary network: edge directions,
    duplicates and values are discarded.

    Parameters
    ----------
    A : scipy.sparse matrix
        Adjacency matrix. Passing a (n_pairs x 2) array of node pairs
        returns the sorted unique pairs as before.
    out_dtype : np.dtype, optional
        Data type of the output matrix, by default np.int8.

    Returns
    -------
    A_uniq : scipy.sparse.csr_matrix
        Symmetric binary CSR adjacency matrix.
    """
    if not issparse(A):
        _warn_legacy_pairs('remove_duplicate_pairs')
        return np.unique(np.sort(A, axis=1), axis=0)
    warnings.warn(
        "`remove_duplicate_pairs` is deprecated, use `to_undirected` instead "
        "(CSR networks built by tysserand have no duplicate edges).",
        DeprecationWarning, stacklevel=2)
    return to_undirected(A, out_dtype=out_dtype)


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


def csr_to_edge_pairs(A: csr_matrix, directed: bool = None) -> np.ndarray:
    """
    Extract the edge pairs of a CSR adjacency matrix.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix.
    directed : bool, optional
        If False, ``A`` is considered symmetric and each edge is returned
        once, with source <= target (upper triangle). If True, every stored
        entry ``A[i, j]`` is returned as an edge from i to j. By default
        (None), it is True if ``A`` is not symmetric (see :func:`is_directed`),
        so that no edge is lost.

    Returns
    -------
    pairs : np.ndarray
        Array of shape (n_edges, 2) of [source, target] node pairs, sorted by
        source then target. Self-loops are kept.
    """
    if directed is None:
        directed = is_directed(A)
    A = _coalesce(A)  # canonical row-major order, no duplicates nor zeros
    rows, cols = A.nonzero()
    if not directed:
        mask = rows <= cols
        rows, cols = rows[mask], cols[mask]
    return np.vstack([rows, cols]).T


def csr_to_edge_coords(coords: np.ndarray, A: csr_matrix, directed: bool = None) -> list:
    """
    Convert a CSR adjacency matrix to a list of edge coordinate arrays
    suitable for a napari Shapes layer.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates in napari convention (row, col order), shape (n_nodes, ndim).
        These should already have been flipped by :func:`convert_nodes_tys_to_nap`.
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix.
    directed : bool, optional
        Passed to :func:`csr_to_edge_pairs`: if True (default when ``A`` is
        not symmetric), each direction of an edge is returned.

    Returns
    -------
    edges_coords : list of np.ndarray
        List of (2, ndim) arrays, one per edge, giving the coordinates of its
        two endpoints. Self-loops are skipped, they can't be drawn as lines.
    """
    pairs = csr_to_edge_pairs(A, directed=directed)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return [coords[pair] for pair in pairs]


def attributes_to_matrix(nodes: pd.DataFrame, attributes: list[str]) -> np.ndarray:
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


def categorical_to_integer(l: ArrayLike) -> np.ndarray:
    uniq = set(l)
    nb_uniq = len(uniq)
    mapping = dict(zip(uniq, range(nb_uniq)))
    converted = [mapping[x] for x in l]
    return converted


def flatten_categories(nodes: pd.DataFrame, att: list[str]) -> pd.Series:
    # the reverse operation is 
    # nodes = nodes.join(pd.get_dummies(nodes['nodes_class']))
    return nodes.loc[:, att].idxmax(axis=1)


def coords_to_df(coords: np.ndarray, columns: Sequence[str] | None = None) -> pd.DataFrame:
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


def pairs_to_df(A: csr_matrix | np.ndarray, columns: Sequence[str] = ('source', 'target'), weight: str | None = None, directed: bool | None = None) -> pd.DataFrame:
    """
    Convert a network into a dataframe of edges.

    Parameters
    ----------
    A : scipy.sparse matrix
        Adjacency matrix. Passing a (n_pairs x 2) array of node pairs is
        deprecated: its rows are used as they are.
    columns : Index or array-like
        Column labels of the edges nodes. Default is ('source', 'target')
    weight : str, optional
        If given, name of an additional column holding the values of ``A``
        (edge weights, distances...).
    directed : bool, optional
        If False, ``A`` is considered symmetric and each edge is exported
        once, with source <= target. If True, every stored entry ``A[i, j]``
        is exported as an edge from i to j. By default (None), it is True
        if ``A`` is not symmetric (see :func:`is_directed`), so that no edge
        nor weight is lost.

    Returns
    -------
    edges : dataframe
        Edges indicated by the nodes 'source' and 'target' they link, sorted
        by source then target. Self-loops are kept.

    Example
    -------
    >>> A = vectors_to_csr(np.array([[1, 0], [1, 2]]))
    >>> pairs_to_df(A)
       source  target
    0       0       1
    1       1       2
    >>> pairs_to_df(vectors_to_csr_directed(np.array([[1, 0], [1, 2]])))
       source  target
    0       1       0
    1       1       2
    """
    if not issparse(A):
        _warn_legacy_pairs('pairs_to_df')
        return pd.DataFrame(data=A, columns=columns)
    if directed is None:
        directed = is_directed(A)
    A = _coalesce(A)  # canonical row-major order, no duplicates nor zeros
    if not directed:
        A = triu(A, k=0, format='csr')
    A = A.tocoo()
    edges = pd.DataFrame({columns[0]: A.row, columns[1]: A.col})
    if weight is not None:
        edges[weight] = A.data
    return edges


def double_sort(data: np.ndarray, last_var: int = 0) -> np.ndarray:
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
