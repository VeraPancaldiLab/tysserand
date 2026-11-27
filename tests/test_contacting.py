"""Tests of the contacting-areas network reconstruction (sequential and tiled)."""
from unittest import mock

import numpy as np
import pytest

from tysserand import tysserand as ty


@pytest.fixture(scope="module", params=[(600, 600, 12), (300, 500, 10), (700, 200, 8)])
def masks(request):
    sx, sy, nb = request.param
    np.random.seed(1)
    _, masks = ty.make_random_tiles(sx=sx, sy=sy, nb=nb, noise_sigma=10.0)
    return masks.astype(np.int32)


def reference_edges(masks, r):
    """Edges from per-object dilation, the slow but straightforward method."""
    labels = np.unique(masks)
    labels = labels[labels > 0]
    edges = set()
    for i in labels:
        for j in ty.find_neighbors(masks, i, r=r):
            edges.add(tuple(sorted(np.searchsorted(labels, [i, j]).tolist())))
    return edges


@pytest.mark.parametrize("r", [1, 3])
def test_build_contacting_matches_dilation(masks, r):
    A = ty.build_contacting(masks, r=r)
    assert A.dtype == np.int8
    assert (A != A.T).nnz == 0
    assert set(map(tuple, ty.csr_to_edge_pairs(A).tolist())) == reference_edges(masks, r)


def test_build_contacting_many_shared_pixels():
    # 2 objects sharing a long border: hundreds of contacting pixel pairs
    # used to overflow the int8 sum and drop the edge
    masks = np.zeros((300, 300), dtype=np.int32)
    masks[:, :150] = 1
    masks[:, 150:] = 2
    A = ty.build_contacting(masks, r=3)
    assert A.toarray().tolist() == [[0, 1], [1, 0]]


@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("n_splits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("backend", ["loky", "threading"])
def test_build_contacting_parallel_identical(masks, r, n_splits, backend):
    A = ty.build_contacting(masks, r=r)
    with mock.patch.object(ty, "choose_optimal_image_split", return_value=n_splits):
        A_paral = ty.build_contacting_parallel(masks, r=r, n_jobs=2, backend=backend)
    assert A_paral.shape == A.shape
    assert (A != A_paral).nnz == 0


def test_make_tiles_limits():
    im = np.arange(16 * 8).reshape(16, 8)
    assert ty.make_tiles_limits(im, 1) == [[0, 8, 0, 16]]
    assert ty.make_tiles_limits(im, 2) == [[0, 4, 0, 16], [4, 8, 0, 16]]
    assert ty.make_tiles_limits(im, 4, margin=1) == [
        [0, 5, 0, 9], [0, 5, 7, 16], [3, 8, 0, 9], [3, 8, 7, 16]]
    assert len(ty.make_tiles_limits(im, 8)) == 8
    # limits are [xmin, xmax, ymin, ymax], x = columns, y = rows
    tile = ty.extract_tile(im, [0, 4, 8, 16])
    assert tile.shape == (8, 4)
    assert tile[0, 0] == im[8, 0]


def test_build_contacting_nn_links_lonely_objects(masks):
    coords, A = ty.build_contacting_nn(masks, r=1, k=3)
    assert A.shape == (coords.shape[0], coords.shape[0])
    assert np.diff(A.indptr).min() >= 1
