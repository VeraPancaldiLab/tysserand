"""
Bridges to NetworkX, iGraph and AnnData.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .convert import categorical_to_integer
from .distances import distance_neighbors

from typing import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    import anndata as ad
    import igraph as ig
    import networkx as nx

__all__ = [
    'to_NetworkX',
    'to_iGraph',
    'add_to_AnnData',
    'get_from_AnnData',
]


def to_NetworkX(
    A: csr_matrix,
    attributes: pd.DataFrame = None,
) -> nx.Graph:
    """
    Convert a CSR adjacency matrix to a NetworkX graph.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix.
    attributes : pd.DataFrame, optional
        Attributes of nodes to be added in NetworkX. Default is None.

    Returns
    -------
    networkx.Graph
        The converted network.
    """
    import networkx as nx
    G = nx.from_scipy_sparse_array(A)
    if attributes is not None:
        for col in attributes.columns:
            nx.set_node_attributes(G, attributes[col].to_dict(), col)
    return G


def to_iGraph(
    A: csr_matrix,
    attributes: pd.DataFrame = None,
    directed: bool = False,
) -> ig.Graph:
    """
    Convert a CSR adjacency matrix to an igraph graph.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix.
    attributes : pd.DataFrame, optional
        Attributes of nodes to be added. Default is None.
    directed : bool, optional
        If False (default), build an undirected graph from the upper triangle
        of A to avoid duplicate edges. If True, build a directed graph using
        all non-zero entries.

    Returns
    -------
    igraph.Graph
        The converted network.
    """
    import igraph as ig
    from scipy.sparse import triu
    if directed:
        G = ig.Graph.Adjacency(A, mode='directed')
    else:
        G = ig.Graph.Adjacency(triu(A, format='csr'), mode='upper')
    if attributes is not None:
        for col in attributes.columns:
            att = attributes[col].values
            if isinstance(att[0], str):
                att = categorical_to_integer(att)
            G.vs[col] = att
    return G


def add_to_AnnData(
    coords: np.ndarray,
    A: csr_matrix,
    D: csr_matrix = None,
    adata: ad.AnnData | None = None,
    counts: ArrayLike | None = None,
    obs_names: Sequence[str] | None = None,
    var_names: Sequence[str] | None = None,
    return_adata: bool = False,
    network_metadata: dict = None,
) -> ad.AnnData | None:
    """
    Add a CSR network to an AnnData (Scanpy) object, created if not provided.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...).
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix.
    D : scipy.sparse.csr_matrix, optional
        CSR distance matrix. If not provided, computed from `coords` and `A`.
    adata : AnnData object, optional
        If not provided, it is created and returned.
    counts : array-like, optional
        Count matrix for AnnData creation. Ignored if `adata` is provided.
    obs_names : array-like, optional
        Observation names. Ignored if `adata` is provided.
    var_names : array-like, optional
        Variable names. Ignored if `adata` is provided.
    return_adata : bool, optional
        Whether to return the AnnData object. The default is False.
    network_metadata : dict, optional
        Metadata stored in `adata.uns['neighbors']`.

    Returns
    -------
    AnnData or None
        The AnnData object if `return_adata` is True or if it was created.
    """
    if adata is None:
        return_adata = True
        import anndata as ad
        if counts is None:
            counts = np.empty(shape=(len(coords), 1))
        adata = ad.AnnData(csr_matrix(counts))
        if obs_names is not None:
            adata.obs_names = obs_names
        if var_names is not None:
            adata.var_names = var_names

    if D is None:
        D = distance_neighbors(coords, A)

    adata.obsm['spatial'] = coords
    adata.obsp['connectivities'] = A
    adata.obsp['distances'] = D
    if network_metadata is None:
        network_metadata = {'connectivities_key': 'connectivities',
                            'distances_key': 'distances',
                            'params': {'method': 'delaunay',
                                       'metric': 'euclidean',
                                       'edge_trimming': 'percentile 99'}}
    adata.uns['neighbors'] = network_metadata
    if return_adata:
        return adata


def get_from_AnnData(
    adata: ad.AnnData,
    key_connectivities: str = 'connectivities',
    key_distances: str = 'distances',
) -> tuple[np.ndarray, csr_matrix, csr_matrix]:
    """
    Extract network CSR matrices and coordinates from an AnnData object.

    Parameters
    ----------
    adata : AnnData object
        An object dedicated to single-cell data analysis.
    key_connectivities : str, optional
        Key in `adata.obsp` for the adjacency matrix. The default is 'connectivities'.
    key_distances : str, optional
        Key in `adata.obsp` for the distance matrix. The default is 'distances'.

    Returns
    -------
    coords : np.ndarray
        Spatial coordinates from `adata.obsm['spatial']`.
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix.
    D : scipy.sparse.csr_matrix
        CSR distance matrix.
    """
    coords = adata.obsm['spatial']
    A = csr_matrix(adata.obsp[key_connectivities])
    D = csr_matrix(adata.obsp[key_distances])
    return coords, A, D
