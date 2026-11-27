"""
Comparison of reconstructed networks with a ground truth.
"""
import numpy as np
from scipy.sparse import issparse, triu

from .convert import _warn_legacy_pairs, to_undirected

__all__ = [
    'confusion_stats',
    'score_method',
]


def confusion_stats(set_true, set_test):
    """
    Count the true positives, false positives and false
    negatives in a test set with respect to a "true" set. 
    True negatives are not counted.
    """
    true_pos = len(set_true.intersection(set_test))
    false_pos = len(set_test.difference(set_true))
    false_neg = len(set_true.difference(set_test))
    
    return true_pos, false_pos, false_neg


def _pad_to_shape(A, shape):
    """Resize a sparse matrix to a larger shape, extra nodes having no edges."""
    A = A.tocsr(copy=True)
    A.resize(shape)
    return A


def score_method(A_true, A_test):
    """
    Compute the rates of true positives, false positives and false negatives
    of the edges of a reconstructed network with respect to a "true" network.
    Rates are relative to the total number of true positives, false positives
    and false negatives (true negatives are not counted).

    Edges are undirected: their direction and multiplicity are ignored.

    Parameters
    ----------
    A_true : scipy.sparse matrix
        Adjacency matrix of the ground truth network.
    A_test : scipy.sparse matrix
        Adjacency matrix of the reconstructed network. If the networks have
        different shapes, the smaller one is padded with unconnected nodes.
        Passing (n_pairs x 2) arrays of "double sorted" node pairs (see
        `double_sort`) is deprecated.

    Returns
    -------
    true_pos_rate, false_pos_rate, false_neg_rate : float

    Examples
    --------
    >>> A_true = ty.vectors_to_csr(np.array([[3, 4], [5, 6], [7, 8]]), n_nodes=9)
    >>> A_test = ty.vectors_to_csr(np.array([[1, 2], [3, 4], [5, 6]]), n_nodes=9)
    >>> score_method(A_true, A_test)
    (0.5, 0.25, 0.25)
    """
    if not (issparse(A_true) and issparse(A_test)):
        _warn_legacy_pairs('score_method')
        set_true = {tuple(e) for e in A_true}
        set_test = {tuple(e) for e in A_test}
        true_pos, false_pos, false_neg = confusion_stats(set_true, set_test)
    else:
        shape = tuple(np.maximum(A_true.shape, A_test.shape))
        # binary upper triangles: one entry per undirected edge
        A_true = triu(to_undirected(_pad_to_shape(A_true, shape)), k=1, format='csr')
        A_test = triu(to_undirected(_pad_to_shape(A_test, shape)), k=1, format='csr')
        true_pos = A_true.multiply(A_test).nnz
        false_pos = A_test.nnz - true_pos
        false_neg = A_true.nnz - true_pos

    total = true_pos + false_pos + false_neg
    true_pos_rate = true_pos / total
    false_pos_rate = false_pos / total
    false_neg_rate = false_neg / total
    
    return true_pos_rate, false_pos_rate, false_neg_rate
