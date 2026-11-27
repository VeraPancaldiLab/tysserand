# tysserand

Fast and accurate spatial networks reconstruction.

*tysserand* reconstructs spatial networks from spatially resolved omics and
bioimaging data. It provides several reconstruction methods, helpers to choose
their parameters, plotting, and bridges to napari, AnnData / Scanpy, NetworkX
and iGraph. The method is published in
[Bioinformatics](https://doi.org/10.1093/bioinformatics/btab490).

## Installation

```bash
pip install tysserand
pip install "tysserand[all]"   # + napari, NetworkX / iGraph / AnnData bridges, notebooks
```

## Usage

Networks are `scipy.sparse` CSR adjacency matrices:

```python
import tysserand as ty

coords = ty.make_random_nodes(size=500, random_state=0)
A = ty.build_delaunay(coords)          # or build_knn, build_rdn, build_lattice...
fig, ax = ty.plot_network(coords, A)
ty.save_network(A, 'network.npz')
```

Segmented images (label masks) are converted to networks of contacting
objects with [`build_contacting`][tysserand.contacting.build_contacting].

The tutorial notebook is
[`examples/02-tutorial.ipynb`](https://github.com/VeraPancaldiLab/tysserand/blob/main/examples/02-tutorial.ipynb).

## Building this documentation

```bash
uv sync --group docs
uv run mkdocs serve     # or `mkdocs build` for a static site in site/
```
