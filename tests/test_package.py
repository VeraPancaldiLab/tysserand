"""Tests of the package layout: light import, public API and backward compatibility."""
import inspect
import subprocess
import sys

import numpy as np
import pytest

import tysserand
from tysserand import tysserand as ty

HEAVY = ["napari", "cv2", "dask", "skimage", "sklearn", "pynndescent", "numba",
         "matplotlib", "seaborn", "anndata", "networkx", "igraph"]


def test_import_is_light():
    """Importing tysserand must not load heavy libraries: they are paid again
    by every parallel worker process."""
    code = ("import sys, tysserand; from tysserand import tysserand; "
            f"print([m for m in {HEAVY!r} if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]", f"heavy modules loaded at import: {out}"


def public_functions(module):
    return {n for n, o in vars(module).items()
            if inspect.isfunction(o) and not n.startswith("_")}


def test_same_api_in_package_and_legacy_module():
    assert public_functions(tysserand) == public_functions(ty)
    for name in ["build_delaunay", "build_knn", "build_contacting", "vectors_to_csr",
                 "save_network", "load_network", "attributes_to_matrix", "plot_network",
                 "to_iGraph", "add_to_AnnData", "visualize"]:
        assert getattr(ty, name) is getattr(tysserand, name)


def test_add_to_anndata_computes_distances():
    pytest.importorskip("anndata")
    coords = np.array([[0., 0.], [1., 0.], [0., 2.]])
    A = ty.vectors_to_csr(np.array([[0, 1], [0, 2]]), n_nodes=3)
    adata = ty.add_to_AnnData(coords, A, return_adata=True)
    D = adata.obsp["distances"]
    assert D[0, 1] == pytest.approx(1)
    assert D[0, 2] == pytest.approx(2)
