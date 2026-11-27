"""
Network reconstruction from segmented images (contacting areas), with a
tiled parallel version.
"""
import os

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix, issparse

from .build import link_solitaries
from .convert import _warn_legacy_pairs, remove_duplicate_pairs, vectors_to_csr

__all__ = [
    'find_neighbors',
    'build_contacting',
    'mask_val_coord',
    'build_contacting_nn',
    'choose_optimal_image_split',
    'split_range',
    'extend_indices',
    'make_tiles_limits',
    'extract_tile',
    'merge_pairs',
    'build_contacting_parallel',
]


def find_neighbors(masks, i, r=1):
    """
    Find the neighbors of a given mask.
    
    Parameters
    ----------
    masks : array_like
        2D array of integers defining the identity of masks
        0 is background (no object detected)
    i : int
        The mask for which we look for the neighbors.
    r : int
        Radius of search.
        
    Returns
    -------
    neighbors : array
        Neighbors in the vicinity of mask `i` in masks.
    """
    from skimage import morphology
    import cv2 as cv
    
    mask = np.uint8(masks == i)
    # create the border in which we'll look at other masks
    kernel = morphology.disk(r)
    dilated = cv.dilate(mask, kernel, iterations=1)
    dilated = dilated.astype(bool)
    # detect potential touching masks
    neighbors = np.unique(masks[dilated])
    # discard the initial cell id of interest
    neighbors = neighbors[neighbors != i]
    # discard the background value
    return neighbors[neighbors != 0]


def build_contacting(
    masks: np.ndarray,
    r: int = 1,
) -> csr_matrix:
    """
    Build a CSR adjacency matrix from segmented regions that contact each other
    or are within a given distance from each other.

    Uses vectorized pixel-shift comparisons over the disk footprint instead of
    per-label dilation, avoiding Python loops over labels entirely.

    Parameters
    ----------
    masks : np.ndarray
        2D array of integers defining the identity of masks.
        0 is background (no object detected).
    r : int, optional
        Radius of search. The default is 1.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric CSR adjacency matrix of shape (n_labels, n_labels),
        where node indices are contiguous and follow the sorted order of
        unique mask labels, matching the row order of `mask_val_coord`.

    Examples
    --------
    >>> A = build_contacting(masks, r=1)
    >>> coords = mask_val_coord(masks)  # row i aligns with node i in A
    """
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]  # exclude background
    src, tgt = _contacting_label_pairs(masks, r=r)
    return _label_pairs_to_csr(src, tgt, unique_labels)


def _contacting_label_pairs(masks, r=1):
    """
    Mask labels of contacting pixel pairs, within a radius `r`.

    Returns two 1D arrays `src, tgt` of mask labels (not node indices), one
    entry per contacting pixel pair (duplicates included).
    """
    from skimage import morphology
    footprint = morphology.disk(r)
    offsets = np.argwhere(footprint) - r

    src_list = []
    tgt_list = []
    for dy, dx in offsets:
        # Only scan the positive half-plane to avoid duplicate edges
        if dy < 0 or (dy == 0 and dx <= 0):
            continue
        # Array views for the original and shifted positions
        sy_o = slice(max(0, -dy), masks.shape[0] - max(0, dy))
        sx_o = slice(max(0, -dx), masks.shape[1] - max(0, dx))
        sy_s = slice(max(0, dy), masks.shape[0] - max(0, -dy))
        sx_s = slice(max(0, dx), masks.shape[1] - max(0, -dx))

        m_o = masks[sy_o, sx_o]
        m_s = masks[sy_s, sx_s]

        contact = (m_o != m_s) & (m_o > 0) & (m_s > 0)
        if contact.any():
            src_list.append(m_o[contact])
            tgt_list.append(m_s[contact])

    if not src_list:
        empty = np.empty(0, dtype=masks.dtype)
        return empty, empty
    return np.concatenate(src_list), np.concatenate(tgt_list)


def _label_pairs_to_csr(src, tgt, unique_labels):
    """
    Binary symmetric CSR adjacency from pairs of mask labels, with node indices
    following the sorted order of `unique_labels` (background excluded).
    """
    n_labels = len(unique_labels)
    if n_labels == 0:
        return csr_matrix((0, 0), dtype=np.int8)
    if len(src) == 0:
        return csr_matrix((n_labels, n_labels), dtype=np.int8)

    # Remap mask labels to contiguous 0-based indices
    src = np.searchsorted(unique_labels, src)
    tgt = np.searchsorted(unique_labels, tgt)
    # Deduplicate the many pixel contacts per pair of objects *before* building
    # the matrix: summing int8 duplicates overflows (e.g. 459 contacts wrap to a
    # negative value) and used to silently drop edges.
    lo = np.minimum(src, tgt).astype(np.int64)
    hi = np.maximum(src, tgt).astype(np.int64)
    codes = np.unique(lo * n_labels + hi)
    pairs = np.column_stack([codes // n_labels, codes % n_labels])
    return vectors_to_csr(pairs, n_nodes=n_labels, out_dtype=np.int8)


def mask_val_coord(
    masks: np.ndarray,
    as_array: bool = True,
) -> np.ndarray:
    """
    Compute centroid coordinates of all labeled regions in a mask image.

    Rows are returned in the sorted order of unique mask labels, matching
    the contiguous 0-based node indexing used by `build_contacting`.

    Parameters
    ----------
    masks : np.ndarray
        2D (or 3D) array of integers defining the identity of masks.
        0 is background (no object detected).
    as_array : bool, optional
        If True (default), return a numpy array with columns ordered (x, y)
        or (x, y, z). If False, return a DataFrame indexed by mask label.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Centroid coordinates. As an array, shape is (n_labels, n_dims).
    """
    from skimage import measure
    props = measure.regionprops_table(masks, properties=('label', 'centroid'))
    coords = pd.DataFrame.from_dict(props)
    if coords.shape[1] == 3:
        coords.rename(columns={'centroid-1': 'x', 'centroid-0': 'y'}, inplace=True)
    elif coords.shape[1] == 4:
        coords.rename(columns={'centroid-2': 'x', 'centroid-1': 'y', 'centroid-0': 'z'},
                      inplace=True)
    else:
        print('More than 3 detected spatial dimensions, check output column names.')
    coords.index = coords['label']
    coords.drop(columns='label', inplace=True)
    if as_array:
        dim_cols = [c for c in ('x', 'y', 'z') if c in coords.columns]
        # regionprops_table returns labels in sorted order, so the output
        # rows are in the same contiguous 0-based order used by build_contacting.
        return coords.loc[:, dim_cols].values
    return coords


def build_contacting_nn(masks, r=1, k=3):
    """
    Build a network from segmented regions as a mix between
    the contacting areas method, that can output some nodes
    edges, and the nearest neighbors method that will link
    these nodes to their neighbors.

    Parameters
    ----------
    masks : array_like
        2D array of integers defining the identity of masks
        0 is background (no object detected)
    r : int
        Radius of search for the contacting areas method. The default is 1.
    k : int, optional
        Number of nearest neighbors. The default is 3.
   

    Returns
    -------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    A : scipy.sparse.csr_matrix
        Symmetric CSR adjacency matrix, nodes follow the sorted order of
        unique mask labels (rows of `coords`). Nodes with less than `k`
        contacting neighbors are linked to their `k` nearest neighbors.
    """
    A = build_contacting(masks, r=r)
    coords = mask_val_coord(masks)
    A = link_solitaries(coords, A, method='knn', min_neighbors=k, verbose=0)
    return coords, A


def choose_optimal_image_split(im, method='im_size', min_tile_size=360000):
    """
    Compute the optimal number of splits of an image
    to run in parallel a function of each core.

    Parameters
    ----------
    im : array_like
        2D array of integers defining the identity of segmented objects
        0 is background (no object detected)
    method : str, optional
        The method used to define the optimal number of splits.
        The default is 'im_size'.
    min_tile_size : int
        Minimum number of bytes of tiles.
        The default is 360000.

    Returns
    -------
    n_splits : int
        The optimal number of splits.
    
    Example
    -------
    >>> im = np.zeros((1024, 1024), dtype=np.int32)
    >>> n_splits = choose_optimal_image_split(im)
    
    Notes
    -----
    One would ideally consider the number of cores, the size of the image
    and the number of detected objects.
    The number of splits should be essentially driven by the size of
    the image and the number of cores.
    The number of splits shouldn't be superior to the number of cores,
    otherwise some cores will wait for the last tiles to be processed
    by other cores, while increasing the inter-process communication
    by too many splits. 
    Ideally n_splits should be a power of 2 in order to split easily 
    the image.
    """
    
    n_cores = os.cpu_count()
    # number of segmented objects, drop the background value
    n_obj = np.unique(im).size - 1
    
    if method == 'im_size':
        # avoid too many splits if image is not so big
        im_size = im.nbytes # slightly different from sys.getsizeof(im)
        # max power of 2
        max_i = int(np.log2(n_cores)) + 1
        n_splits = 1
        for i in range(1, max_i):
            new_split = 2**i
            if im_size / new_split >= min_tile_size:
                n_splits = new_split
            else:
                break
    elif method == 'naive':
        n_splits = n_cores
    
    return n_splits


def split_range(r, n):
    """
    Computes the indices of segments after splitting a range of r values
    into n segments.

    Parameters
    ----------
    r : int
        Size of the range vector.
    n : int
        The number of splits.

    Returns
    -------
    segments : list
        The list of lists of first and last indices of segments.
    
    Example
    -------
    >>> split_range(8, 2)
    [[0, 4], [4, 8]]
    """
    
    step = int(r / n)
    segments = []
    for i in range(n):
        new_segment = [step * i, step * (i + 1)]
        segments.append(new_segment)
    # correct the gap in the missing index due to the truncated step
    segments[-1][-1] = r
    return segments


def extend_indices(segments, margin):
    """
    Decrease and increase the values of the first and last elements
    respectively in each list of segments by a given margin.
    The first indice of the first segment and the last indice of the
    last segments are not modified.

    Parameters
    ----------
    segments : list
        The list of lists of first and last indices of segments.
    margin : int
        The extra extend to add on each side of segments.
    
    Example
    -------
    >>> segments = split_range(16, 4)
    >>> extend_indices(segments, margin=1)
    [[0, 5], [3, 9], [7, 13], [11, 16]]
    """
    
    if len(segments) == 1:
        return segments
    else:
        # first process the first and last segments
        segments[0][-1] += margin
        segments[-1][0] -= margin
        # if there are more than 2 segments
        for i in range(len(segments))[1:-1]:
            segments[i][0] -= margin
            segments[i][-1] += margin
    return segments


def make_tiles_limits(im, n_splits, margin=0):
    """
    Compute the indices in an image to split it into several tiles.

    Parameters
    ----------
    im : array_like
        2D array of integers defining the identity of segmented objects
        0 is background (no object detected)
    n_splits : int
        The number of splits.
    margin : int
        The extra space to include at the border of tiles.
        The default is 0.

    Returns
    -------
    tiles_indices : list
        The list of indices [[xmin, xmax], [ymin, ymax]] for each tile.
        
    Example
    -------
    >>> im = np.arange(16 * 8).reshape(16, 8)
    >>> make_tiles_limits(im, 4, margin=0)
    [[0, 4, 0, 8], [0, 4, 8, 16], [4, 8, 0, 8], [4, 8, 8, 16]]
    >>> make_tiles_limits(im, 4, margin=1)
    [[0, 5, 0, 9], [0, 5, 7, 16], [3, 8, 0, 9], [3, 8, 7, 16]]
    >>> make_tiles_limits(im, 1)
    [[0, 8, 0, 16]]
    """
    
    if n_splits <= 1:
        return [[0, im.shape[1], 0, im.shape[0]]]
    # number of splits per axis, with nx * ny == n_splits for powers of 2
    nx = 2 ** int(np.ceil(np.log2(n_splits) / 2))
    ny = max(1, n_splits // nx)
    x_segments = split_range(im.shape[1], nx)
    y_segments = split_range(im.shape[0], ny)
    
    if margin > 0:
        x_segments = extend_indices(x_segments, margin=margin)
        y_segments = extend_indices(y_segments, margin=margin)
    
    # make combinations of [xmin, xmax, ymin, ymax] indices of tiles
    tiles_indices = []
    for xlim in x_segments:
        for ylim in y_segments:
            tiles_indices.append(xlim + ylim)
    return tiles_indices


def extract_tile(im, limits):
    """
    Extract a tile from an image given
    its [xmin, xmax, ymin, ymax] limit indices.

    Parameters
    ----------
    im : array_like
        2D array of integers defining the identity of segmented objects
        0 is background (no object detected)
    limits : list
        The list of limit indices [xmin, xmax, ymin, ymax].

    Returns
    -------
    tile : array_like
        The extracted tile.
        
    Example
    -------
    >>> im = np.arange(8 * 8).reshape(8, 8)
    >>> tiles_indices = make_tiles_limits(im, 4, margin=0)
    >>> extract_tile(im, tiles_indices[-1])
    array([[36, 37, 38, 39],
           [44, 45, 46, 47],
           [52, 53, 54, 55],
           [60, 61, 62, 63]])
    """
    
    # limits are [xmin, xmax, ymin, ymax], x being columns and y rows
    tile = im[limits[2]: limits[3], limits[0]: limits[1]]
    return tile


def merge_pairs(lA, out_dtype=np.int8):
    """
    Merge a list of networks into a single network, the union of their edges.

    Parameters
    ----------
    lA : list
        The list of networks as sparse adjacency matrices of identical shape.
        Passing a list of Nx2 arrays of node pairs is deprecated: it returns
        their unique rows as before.
    out_dtype : np.dtype, optional
        Data type of the output matrix, by default np.int8.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Symmetric binary CSR adjacency matrix of the merged edges.

    Example
    -------
    >>> A1 = vectors_to_csr(np.array([[0, 1], [2, 3]]), n_nodes=6)
    >>> A2 = vectors_to_csr(np.array([[2, 3], [4, 5]]), n_nodes=6)
    >>> csr_to_edge_pairs(merge_pairs([A1, A2]))
    array([[0, 1],
           [2, 3],
           [4, 5]])
    """
    if not all(issparse(A) for A in lA):
        _warn_legacy_pairs('merge_pairs')
        return np.unique(np.vstack(lA), axis=0)
    shapes = {A.shape for A in lA}
    if len(shapes) != 1:
        raise ValueError(f"networks have different shapes: {shapes}, "
                         "build them with the same `n_nodes`")
    A = csr_matrix(lA[0]).astype(bool)
    for B in lA[1:]:
        A = A + csr_matrix(B).astype(bool)  # logical OR
    return remove_duplicate_pairs(A, out_dtype=out_dtype)


def build_contacting_parallel(
    im: np.ndarray,
    r: int = 1,
    split_method: str = 'im_size',
    min_tile_size: int = 360000,
    n_jobs: int = -1,
    backend: str = 'loky',
) -> csr_matrix:
    """
    Build a CSR adjacency matrix from segmented regions that contact each other
    or are within a given distance from each other, processing tiles of the
    image in parallel with joblib.

    The image is split into overlapping tiles (margin `r`) so that contacts
    across tile borders are detected, and the output is identical to
    `build_contacting`.

    Parameters
    ----------
    im : np.ndarray
        2D array of integers defining the identity of masks
        0 is background (no object detected)
    r : int
        Radius of search.
    split_method : str, optional
        The method used to define the optimal number of splits.
        The default is 'im_size'.
    min_tile_size : int
        Minimum number of bytes of tiles.
        The default is 360000.
    n_jobs : int, optional
        Number of joblib workers, -1 uses all CPUs. The default is -1.
    backend : str, optional
        joblib backend, 'loky' (processes) or 'threading'. The default is 'loky'.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric CSR adjacency matrix of shape (n_labels, n_labels),
        where node indices are contiguous and follow the sorted order of
        unique mask labels, matching the row order of `mask_val_coord`.

    Example
    -------
    >>> coords, masks = ty.make_random_tiles(sx=600, sy=600, nb=12, noise_sigma=10.0)
    >>> A = ty.build_contacting(masks)
    >>> A_paral = ty.build_contacting_parallel(masks)
    >>> (A != A_paral).nnz == 0
    True
    """
    unique_labels = np.unique(im)
    unique_labels = unique_labels[unique_labels > 0]  # exclude background

    n_splits = choose_optimal_image_split(im, method=split_method, min_tile_size=min_tile_size)
    tiles_limits = make_tiles_limits(im, n_splits, margin=r)

    if len(tiles_limits) == 1:
        lpairs = [_contacting_label_pairs(im, r=r)]
    else:
        n_jobs = min(len(tiles_limits), joblib.cpu_count() if n_jobs == -1 else n_jobs)
        lpairs = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(_contacting_label_pairs)(extract_tile(im, limits), r=r)
            for limits in tiles_limits
        )
    # pairs are mask labels, hence global across tiles: merge them directly
    src = np.concatenate([p[0] for p in lpairs])
    tgt = np.concatenate([p[1] for p in lpairs])
    return _label_pairs_to_csr(src, tgt, unique_labels)
