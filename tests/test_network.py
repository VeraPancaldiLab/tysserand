"""Tests of the CSR network representation: reconstruction methods, conversions,
IO and the CSR ports of the former edge-list functions."""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import tysserand as ty


def edge_set(A):
    return set(map(tuple, ty.csr_to_edge_pairs(A).tolist()))


def assert_symmetric_binary(A, n_nodes):
    assert isinstance(A, csr_matrix)
    assert A.shape == (n_nodes, n_nodes)
    assert A.dtype == np.int8
    assert (A != A.T).nnz == 0
    assert set(np.unique(A.data).tolist()) <= {1}


@pytest.fixture(scope="module")
def coords():
    return ty.make_random_nodes(size=200, ndim=2)


# ------ reconstruction ------

@pytest.mark.parametrize("build", [
    lambda c: ty.build_delaunay(c),
    lambda c: ty.build_delaunay(c, node_adaptive_trimming=True),
    lambda c: ty.build_knn(c, k=5),
    lambda c: ty.build_rdn(c, r=0.1),
])
def test_build_returns_symmetric_csr(coords, build):
    A = build(coords)
    assert_symmetric_binary(A, len(coords))
    assert A.diagonal().sum() == 0
    assert A.nnz > 0


def test_build_3d():
    coords = ty.make_random_nodes(size=100, ndim=3)
    assert_symmetric_binary(ty.build_delaunay(coords), len(coords))


def test_build_knn_min_degree(coords):
    A = ty.build_knn(coords, k=5)
    assert np.diff(A.indptr).min() >= 5


def test_build_rdn_matches_bruteforce(coords):
    r = 0.1
    A = ty.build_rdn(coords, r=r)
    dist = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    expected = {(i, j) for i, j in zip(*np.nonzero(dist <= r)) if i < j}
    assert edge_set(A) == expected


def test_build_delaunay_return_dist(coords):
    A, D = ty.build_delaunay(coords, return_dist=True)
    assert (A != 0).sum() == (D != 0).sum()
    i, j = ty.csr_to_edge_pairs(A)[0]
    assert D[i, j] == pytest.approx(np.linalg.norm(coords[i] - coords[j]))


# ------ conversions ------

def test_vectors_csr_roundtrip():
    pairs = np.array([[0, 1], [1, 2], [0, 3]])
    A = ty.vectors_to_csr(pairs, n_nodes=5)  # node 4 is isolated
    assert_symmetric_binary(A, 5)
    assert edge_set(A) == {(0, 1), (1, 2), (0, 3)}
    B = ty.vectors_to_csr_directed(pairs, n_nodes=5)
    assert (B != B.T).nnz > 0
    assert B[0, 1] == 1 and B[1, 0] == 0


def test_distance_neighbors():
    coords = np.array([[0., 0.], [3., 4.], [0., 1.]])
    A = ty.vectors_to_csr(np.array([[0, 1], [0, 2]]))
    D = ty.distance_neighbors(coords, A)
    assert D[0, 1] == pytest.approx(5) and D[1, 0] == pytest.approx(5)
    assert D[0, 2] == pytest.approx(1)
    assert D[1, 2] == 0


def test_attributes_to_matrix():
    nodes = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [5, 6, 7]})
    C = ty.attributes_to_matrix(nodes, ['a', 'b'])
    assert C.shape == (3, 2) and C.dtype == float
    assert ty.attributes_to_matrix(nodes, 'c').shape == (3, 1)


def directed_weighted():
    # 0 -> 1 (0.5), 1 -> 0 (2.), 1 -> 2 (3.), 3 -> 2 (4.)
    rows, cols = np.array([0, 1, 1, 3]), np.array([1, 0, 2, 2])
    return csr_matrix((np.array([0.5, 2., 3., 4.]), (rows, cols)), shape=(4, 4))


@pytest.mark.parametrize("build", [
    lambda c: ty.build_delaunay(c),
    lambda c: ty.build_delaunay(c, node_adaptive_trimming=True),
    lambda c: ty.build_knn(c, k=6),
    lambda c: ty.build_rdn(c, r=0.1),
])
def test_built_networks_are_already_undirected(coords, build):
    # networks built by tysserand need no deduplication nor symmetrization
    A = build(coords)
    assert_symmetric_binary(A, len(coords))
    assert not ty.is_directed(A)
    assert (ty.to_undirected(A) != A).nnz == 0
    assert (ty.to_undirected(A, weighted=True) != A).nnz == 0


def test_is_directed():
    assert ty.is_directed(ty.vectors_to_csr_directed(np.array([[0, 1]]), n_nodes=2))
    assert not ty.is_directed(ty.vectors_to_csr(np.array([[0, 1]]), n_nodes=2))
    assert ty.is_directed(directed_weighted())  # asymmetric weights too
    # non-coalesced duplicates of a symmetric network
    A = csr_matrix((np.ones(4), (np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))), shape=(2, 2))
    assert not ty.is_directed(A)


def test_to_undirected_binary():
    # duplicated, directed and opposite entries, plus an explicit zero
    rows = np.array([4, 1, 3, 2, 1, 0])
    cols = np.array([3, 2, 4, 1, 2, 4])
    data = np.array([1, 1, 1, 1, 1, 0], dtype=np.int8)
    A = csr_matrix((data, (rows, cols)), shape=(5, 5))
    B = ty.to_undirected(A)
    assert_symmetric_binary(B, 5)
    assert edge_set(B) == {(1, 2), (3, 4)}


def test_to_undirected_no_overflow():
    # many duplicates of an int8 edge used to overflow when summed
    pairs = np.repeat([[0, 1]], 300, axis=0)
    A = csr_matrix((np.ones(300, dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(2, 2))
    assert ty.to_undirected(A).toarray().tolist() == [[0, 1], [1, 0]]


def test_to_undirected_weighted():
    A = directed_weighted()
    B = ty.to_undirected(A, weighted=True)
    assert B.dtype == float and not ty.is_directed(B)
    assert B.toarray().tolist() == [[0, 2, 0, 0], [2, 0, 3, 0], [0, 3, 0, 4], [0, 0, 4, 0]]
    B = ty.to_undirected(A, weighted=True, combine='sum')
    assert B[0, 1] == B[1, 0] == 2.5 and B[1, 2] == 3
    with pytest.raises(ValueError):
        ty.to_undirected(A, weighted=True, combine='mean')


def test_remove_duplicate_pairs_is_deprecated():
    A = ty.vectors_to_csr_directed(np.array([[4, 3], [1, 2], [3, 4], [2, 1]]))
    with pytest.deprecated_call():
        B = ty.remove_duplicate_pairs(A)
    assert (B != ty.to_undirected(A)).nnz == 0


def test_merge_pairs_csr():
    A1 = ty.vectors_to_csr(np.array([[0, 1], [2, 3]]), n_nodes=6)
    A2 = ty.vectors_to_csr(np.array([[3, 2], [4, 5]]), n_nodes=6)
    A = ty.merge_pairs([A1, A2])
    assert_symmetric_binary(A, 6)
    assert edge_set(A) == {(0, 1), (2, 3), (4, 5)}
    with pytest.raises(ValueError):
        ty.merge_pairs([A1, ty.vectors_to_csr(np.array([[0, 1]]), n_nodes=2)])


def test_merge_pairs_keeps_directions():
    A1 = ty.vectors_to_csr_directed(np.array([[0, 1], [2, 1]]), n_nodes=3)
    A2 = ty.vectors_to_csr_directed(np.array([[0, 1], [1, 2]]), n_nodes=3)
    A = ty.merge_pairs([A1, A2])
    assert A.dtype == np.int8
    assert A.toarray().tolist() == [[0, 1, 0], [0, 0, 1], [0, 1, 0]]


def test_merge_pairs_weighted():
    A1 = directed_weighted()
    A2 = csr_matrix((np.array([1., 5.]), (np.array([0, 2]), np.array([1, 3]))), shape=(4, 4))
    A = ty.merge_pairs([A1, A2], weighted=True)
    assert A[0, 1] == 1 and A[1, 0] == 2 and A[2, 3] == 5 and A[3, 2] == 4
    assert ty.is_directed(A)
    A = ty.merge_pairs([A1, A2], weighted=True, combine='sum')
    assert A[0, 1] == 1.5 and A[1, 2] == 3
    # binary union by default
    assert set(np.unique(ty.merge_pairs([A1, A2]).data).tolist()) == {1}


def test_pairs_to_df_csr():
    A = ty.vectors_to_csr(np.array([[2, 1], [0, 1], [0, 3]]))
    edges = ty.pairs_to_df(A)
    assert edges.columns.tolist() == ['source', 'target']
    assert edges.values.tolist() == [[0, 1], [0, 3], [1, 2]]
    D = ty.distance_neighbors(np.array([[0., 0], [1, 0], [1, 2], [0, 3]]), A)
    edges = ty.pairs_to_df(D, columns=['i', 'j'], weight='distance')
    assert edges['distance'].tolist() == pytest.approx([1, 3, 2])


def test_pairs_to_df_directed():
    D = ty.vectors_to_csr_directed(np.array([[2, 1], [0, 1]]), n_nodes=3)
    # edges of the lower triangle used to be dropped
    assert ty.pairs_to_df(D).values.tolist() == [[0, 1], [2, 1]]
    edges = ty.pairs_to_df(directed_weighted(), weight='w')
    assert edges.values.tolist() == [[0, 1, 0.5], [1, 0, 2], [1, 2, 3], [3, 2, 4]]
    # forcing directed=False exports the upper triangle only
    assert ty.pairs_to_df(D, directed=False).values.tolist() == [[0, 1]]
    # a symmetric network exported as directed has both directions
    A = ty.vectors_to_csr(np.array([[0, 1]]), n_nodes=2)
    assert ty.pairs_to_df(A, directed=True).values.tolist() == [[0, 1], [1, 0]]


def test_pairs_to_df_self_loops():
    A = csr_matrix((np.ones(3, dtype=np.int8), (np.array([0, 0, 1]), np.array([0, 1, 0]))), shape=(2, 2))
    assert ty.pairs_to_df(A).values.tolist() == [[0, 0], [0, 1]]



def test_csr_to_edge_pairs_directed():
    D = ty.vectors_to_csr_directed(np.array([[2, 1], [0, 1], [1, 0]]), n_nodes=3)
    # edges of the lower triangle used to be dropped
    assert ty.csr_to_edge_pairs(D).tolist() == [[0, 1], [1, 0], [2, 1]]
    assert ty.csr_to_edge_pairs(D, directed=False).tolist() == [[0, 1]]
    A = ty.vectors_to_csr(np.array([[2, 1], [0, 1]]), n_nodes=3)
    assert ty.csr_to_edge_pairs(A).tolist() == [[0, 1], [1, 2]]
    assert ty.csr_to_edge_pairs(A, directed=True).tolist() == [[0, 1], [1, 0], [1, 2], [2, 1]]
    # same edges as pairs_to_df
    assert ty.csr_to_edge_pairs(D).tolist() == ty.pairs_to_df(D).values.tolist()


def test_csr_to_edge_pairs_duplicates_and_self_loops():
    # non-coalesced duplicates, an explicit zero and a self-loop
    rows, cols = np.array([0, 0, 1, 1, 2, 2]), np.array([1, 1, 0, 0, 2, 0])
    data = np.array([1, 1, 1, 1, 1, 0], dtype=np.int8)
    A = csr_matrix((data, (rows, cols)), shape=(3, 3))
    assert ty.csr_to_edge_pairs(A).tolist() == [[0, 1], [2, 2]]
    coords = np.array([[0., 0], [1, 0], [2, 0]])
    edges_coords = ty.csr_to_edge_coords(coords, A)  # self-loop not drawable
    assert len(edges_coords) == 1 and edges_coords[0].tolist() == [[0, 0], [1, 0]]


def test_score_method_csr():
    A_true = ty.vectors_to_csr(np.array([[3, 4], [5, 6], [7, 8]]), n_nodes=9)
    A_test = ty.vectors_to_csr(np.array([[2, 1], [4, 3], [5, 6]]), n_nodes=7)
    assert ty.score_method(A_true, A_test) == (0.5, 0.25, 0.25)
    assert ty.score_method(A_true, A_true) == (1, 0, 0)


@pytest.mark.parametrize("fct, args", [
    (ty.remove_duplicate_pairs, ([[4, 3], [1, 2], [3, 4], [2, 1]],)),
    (ty.merge_pairs, ([np.array([[0, 1]]), np.array([[0, 1], [2, 3]])],)),
    (ty.pairs_to_df, (np.array([[0, 1], [1, 2]]),)),
    (ty.score_method, (np.array([[3, 4], [5, 6]]), np.array([[3, 4]]))),
])
def test_edge_list_input_is_deprecated(fct, args):
    with pytest.deprecated_call():
        fct(*args)


def test_legacy_edge_list_results():
    with pytest.deprecated_call():
        assert ty.remove_duplicate_pairs([[4, 3], [1, 2], [3, 4], [2, 1]]).tolist() == [[1, 2], [3, 4]]
    with pytest.deprecated_call():
        assert ty.score_method(np.array([[3, 4], [5, 6], [7, 8]]),
                               np.array([[1, 2], [3, 4], [5, 6]])) == (0.5, 0.25, 0.25)


# ------ link_solitaries ------

@pytest.mark.parametrize("method", ["knn", "delaunay"])
def test_link_solitaries(method):
    # 5 collinear nodes; only 0-1 linked, nodes 2, 3 and 4 are solitary
    coords = np.array([[0., 0], [1, 0], [2, 0], [3.1, 0], [4, 0.1]])
    A = ty.vectors_to_csr(np.array([[0, 1]]), n_nodes=5)
    B = ty.link_solitaries(coords, A, method=method, min_neighbors=1, verbose=0)
    assert_symmetric_binary(B, 5)
    assert np.diff(B.indptr).min() >= 1  # no solitary node remains
    assert (0, 1) in edge_set(B)


def test_link_solitaries_missing_trailing_nodes():
    # A built without n_nodes: trailing isolated nodes are not in the matrix
    coords = np.array([[0., 0], [1, 0], [2, 0], [3, 0]])
    A = ty.vectors_to_csr(np.array([[0, 1]]))
    B = ty.link_solitaries(coords, A, method='knn', min_neighbors=1, verbose=0)
    assert B.shape == (4, 4)
    assert np.diff(B.indptr).min() >= 1


def test_link_solitaries_min_neighbors():
    coords = ty.make_random_nodes(size=50, ndim=2)
    A = ty.build_delaunay(coords)
    B = ty.link_solitaries(coords, A, method='knn', min_neighbors=4, verbose=0)
    assert np.diff(B.indptr).min() >= 4
    assert edge_set(A) <= edge_set(B)



def test_link_solitaries_keeps_directions_and_weights():
    coords = np.array([[0., 0], [1, 0], [2, 0], [3.1, 0], [4, 0.1]])
    # directed, weighted edge 1 -> 0: nodes 0 and 1 aren't solitary
    A = csr_matrix((np.array([0.5]), (np.array([1]), np.array([0]))), shape=(5, 5))
    B = ty.link_solitaries(coords, A, method='knn', min_neighbors=1, verbose=0)
    assert B.dtype == float
    assert B[1, 0] == 0.5 and B[0, 1] == 0  # existing edge unchanged
    # only solitary nodes 2, 3 and 4 get new undirected edges, of value 1
    new = B - A
    new.eliminate_zeros()
    assert not ty.is_directed(new)
    assert set(np.unique(new.data).tolist()) == {1}
    assert set(new.nonzero()[0].tolist()) >= {2, 3, 4}
    assert (ty.to_undirected(B).indptr[1:] - ty.to_undirected(B).indptr[:-1]).min() >= 1


def test_link_solitaries_existing_edge_not_doubled():
    # node 1 has 1 neighbor and needs 2: its nearest neighbor 0 is already linked
    coords = np.array([[0., 0], [1, 0], [3, 0]])
    A = ty.vectors_to_csr(np.array([[0, 1]]), n_nodes=3)
    B = ty.link_solitaries(coords, A, method='knn', min_neighbors=2, verbose=0)
    assert_symmetric_binary(B, 3)


# ------ IO ------

def test_save_load_roundtrip(tmp_path, coords):
    A = ty.build_delaunay(coords)
    p = tmp_path / "edges_test.npz"
    ty.save_network(A, p)
    B = ty.load_network(p)
    assert isinstance(B, csr_matrix)
    assert (A != B).nnz == 0


@pytest.mark.parametrize("extension", ["parquet", "csv"])
def test_migrate_edges_to_npz(tmp_path, extension):
    if extension == "parquet":
        pytest.importorskip("pyarrow")
    coords = ty.make_random_nodes(size=30, ndim=2)
    # the last node is isolated: n_nodes must come from the nodes file
    A = ty.build_delaunay(coords[:-1])
    A.resize((30, 30))
    nodes = ty.coords_to_df(coords)
    edges = ty.pairs_to_df(A)
    if extension == "parquet":
        nodes.to_parquet(tmp_path / "nodes_patient-1_sample-1.parquet", index=False)
        edges.to_parquet(tmp_path / "edges_patient-1_sample-1.parquet", index=False)
    else:
        nodes.to_csv(tmp_path / "nodes_patient-1_sample-1.csv", index=False)
        edges.to_csv(tmp_path / "edges_patient-1_sample-1.csv", index=False)
    ty.migrate_edges_to_npz(tmp_path, extension=extension, verbose=0)
    B = ty.load_network(tmp_path / "edges_patient-1_sample-1.npz")
    assert B.shape == (len(nodes), len(nodes))
    assert (B != A).nnz == 0
