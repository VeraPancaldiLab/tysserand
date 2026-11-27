# Changelog

## v1.0.0 — 2026-11-27

First stable release: networks are `scipy.sparse` CSR adjacency matrices throughout, the API follows semantic versioning from now on (breaking changes only in major releases).

### Added
  - `to_undirected(A, weighted=False, combine='max')`: symmetrize a network, binary or keeping edge values
  - `is_directed(A)`: whether an adjacency matrix is asymmetric
  - type hints on the whole public API
  - API documentation site (mkdocs + mkdocstrings): `uv sync --group docs`, then `uv run mkdocs serve`
  - `random_state` argument of `make_random_nodes`, `make_random_tiles` and `build_knn` (approximate search) for reproducible results; they no longer use numpy's global random state

### Changed
  - `remove_duplicate_pairs`, `merge_pairs`, `pairs_to_df` and `score_method` take CSR adjacency matrices; edge-list arrays are deprecated (`DeprecationWarning`)
  - `remove_duplicate_pairs` is deprecated in favor of `to_undirected` (CSR networks have no duplicate edges)
  - `merge_pairs` keeps edge directions, and edge values with `weighted=True` (`combine='max'` or `'sum'`)
  - `pairs_to_df` can export edge values (weights, distances) with `weight='column_name'`, exports every edge of directed networks (`directed=None` detects them, previously edges of the lower triangle were dropped) and keeps self-loops
  - `csr_to_edge_pairs` and `csr_to_edge_coords` return every edge of directed networks (`directed=None` detects them, previously edges of the lower triangle were dropped); `csr_to_edge_pairs` keeps self-loops and ignores non-coalesced duplicates
  - `link_solitaries` keeps the values and directions of existing edges (they were binarized and symmetrized); new edges are undirected with a value of 1
  - packaging: single PEP 621 `pyproject.toml` managed with uv (Poetry and `setup.py` removed); requires Python >= 3.10; `napari`, `ipykernel`, `qtconsole` and `pathlib` are no longer core dependencies (`napari` is an extra, along with `interop`, `notebooks` and `all`); unused `hdbscan` and `umap-learn` removed
  - code split into submodules (`build`, `contacting`, `convert`, `distances`, `io`, `plot`, `interop`, `annotation`, `scoring`, `generators`), all public functions available from `import tysserand as ty`
  - heavy libraries (napari, OpenCV, scikit-image, scikit-learn, pynndescent, matplotlib, seaborn) are imported only when used: `import tysserand` drops from ~1.4 s / ~360 MB to ~0.3 s / ~130 MB
  - `build_contacting_parallel` uses joblib instead of Dask (Dask is no longer a dependency) and returns a CSR adjacency matrix identical to `build_contacting`

### Development
  - code linted and formatted with ruff (`uv run ruff check .`, `uv run ruff format .`), pre-commit hooks (`uv run pre-commit install`) and GitHub Actions running the linters, the tests (Python 3.10 and 3.12) and the documentation build

### Deprecated
  - `from tysserand import tysserand as ty`: use `import tysserand as ty`; the `tysserand.tysserand` module still re-exports the whole API but emits a `FutureWarning` and will be removed in v2.0

### Fixed
  - `link_solitaries` with a network missing trailing isolated nodes, or with self-loops
  - `score_method` docstring: it returns 3 rates
  - tiling of `build_contacting_parallel` (1 or 2 splits, non power-of-4 splits, non-square images)
  - `build_contacting_nn` with CSR networks
  - `add_to_AnnData` failing when distances `D` are not provided
  - `choose_optimal_image_split` no longer computes the unused number of objects of the whole image
  - `build_contacting` silently dropped edges between objects sharing many contacting pixels (int8 overflow when summing duplicate contacts)

## v0.6.0

### Added
  - `get_from_AnnData` function
  - option to link nodes of 2 different sets in `build_rdn`
  - `node_adaptive_trimming` reconstruction method
  - `delaunay` method to `link_solitaries` function
  - `build_lattice` reconstruction method
  - use pynndescent for larger network reconstruction with knn

### Changed
  - upgrade compatibility with AnnData

### Fixed
  - fix legend in plot_network functions

## v0.5.0
In v0.5.0 3D network reconstruction is officially supported. The core functions were actually already compatible with n-dimensional networks. Here we added additional support in some data wrangling functions, and a tutorial to show what 3D nets look like in Napari.

## v0.4.0
v0.4.0 adds functions to help using Napari for interactive visualization and annotation, which comes with an additional tutorial. Plus some code refactoring and more detailed instructions in tutorials and README.

## v0.3.0

v0.3.0 implements 2 new methods, corrects a bug and improves the project organization.

**New features**

- The `build_contacting_nn` method allows to link segmented objects that are close to each other, then to link the remaining "lonely" objects to their nearest neighbors in order to avoid too many unconnect nodes in the resulting network
- The `build_contacting` method has a parallel version implemented with Dask, in which big images are splitted into an optimal numberof tiles depending on the number of cores.

**Fixed bugs:**

- The `build_knn` method was actually building networks with (k-1) neighbors as the first k was 'oneself' and was discarded. Now it build networks with k neighboring nodes.

**Project organization**

- Added poetry project configuration files `pyproject.toml` and `poetry.lock` for better reproducibility, see [here](https://modelpredict.com/python-dependency-management-tools) for some discussion. A guide for easy installation with all commands coming soon :)
- This very CHANGELOG file!


## v0.2.0

v0.2.0 implement a new method for more accurate network reconstruction, makes the `build_contacting` method faster, and add the first functions to help using Napari as an ineractive tool to add and modify network annotations (tutorial coming soon).

**New features**

- The `build_contacting` method is 19 times faster thanks to OpenCV!
- New `percentile_size` method to find the optimal trimming distance for Delaunay triangulation. It is based on the proportion of border edges given the number of nodes, and the proportion of artifactual edges in these border nodes. It was shown to perform very well on realistic simulated images, and it was the best performing method on a real bioimage, see the [publcation](https://doi.org/10.1093/bioinformatics/btab490)
- Add a set of utilities to help using the Napari interactive image visualization library to add and modify nodes and edges annotations, clean edges data while being able to pan, zoom and hide or vary image channels intensities.

**Fixed bugs:**

- The `build_contacting` method was not compatibe with images with some missing objects, like segmented objects deleted after filtering by size or other criteria. Now it's solved.
