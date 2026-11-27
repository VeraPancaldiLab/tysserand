"""
Network persistence as sparse ``.npz`` files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from .convert import vectors_to_csr

__all__ = [
    'save_network',
    'load_network',
    'migrate_edges_to_npz',
]


def save_network(A: csr_matrix, path: str | Path) -> None:
    """
    Save a CSR adjacency matrix to a sparse ``.npz`` file.

    Parameters
    ----------
    A : scipy.sparse matrix
        Adjacency matrix of the network. Coerced to CSR if needed.
    path : str or Path
        Output path. The ``.npz`` extension is added by scipy if absent.
    """
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    save_npz(str(path), A)


def load_network(path: str | Path) -> csr_matrix:
    """
    Load a CSR adjacency matrix saved with :func:`save_network`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        The network's adjacency matrix.
    """
    return load_npz(str(path)).tocsr()


def migrate_edges_to_npz(
    directory: str | Path,
    nodes_dir: str | Path | None = None,
    extension: str = 'parquet',
    overwrite: bool = False,
    verbose: int = 1,
) -> None:
    """
    One-time migration of edge-list network files to sparse ``.npz`` CSR files.

    Reads each ``edges_{group}.{extension}`` file (first two columns are the
    ``source``/``target`` node indices), infers the number of nodes from the
    matching ``nodes_{group}`` file (to keep isolated nodes), converts to a
    symmetric CSR adjacency matrix with :func:`vectors_to_csr` and writes
    ``edges_{group}.npz`` next to it.

    Parameters
    ----------
    directory : str or Path
        Directory holding the ``edges_*`` files (and ``nodes_*`` files if
        ``nodes_dir`` is not given).
    nodes_dir : str or Path, optional
        Directory holding the ``nodes_*`` files, if different from ``directory``.
    extension : str, optional
        Extension of the existing edge-list files, ``'parquet'`` (default) or ``'csv'``.
    overwrite : bool, optional
        If False (default), skip groups whose ``.npz`` file already exists.
    verbose : int, optional
        Verbosity level.
    """
    directory = Path(directory)
    nodes_dir = directory if nodes_dir is None else Path(nodes_dir)
    if extension == 'parquet':
        read_fct = pd.read_parquet
    elif extension == 'csv':
        read_fct = pd.read_csv
    else:
        raise ValueError(f"Unsupported extension: {extension}")

    for edges_path in sorted(directory.glob(f'edges_*.{extension}')):
        group = edges_path.stem[len('edges_') :]
        out_path = directory / f'edges_{group}.npz'
        if out_path.exists() and not overwrite:
            continue
        pairs = read_fct(edges_path).iloc[:, :2].to_numpy().astype(int)
        # infer n_nodes from the matching nodes file to preserve isolated nodes
        n_nodes = None
        nodes_path = nodes_dir / f'nodes_{group}.{extension}'
        if nodes_path.exists():
            n_nodes = len(read_fct(nodes_path))
        A = vectors_to_csr(pairs, n_nodes=n_nodes)
        save_network(A, out_path)
        if verbose > 0:
            print(f"migrated {edges_path.name} -> {out_path.name}")
