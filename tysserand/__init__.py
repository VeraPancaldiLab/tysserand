#! /usr/bin/env python3

"""
Python Spatial Network Construction
===============================================

Reconstruct spatial networks, as symmetric ``scipy.sparse`` CSR adjacency
matrices, from node coordinates or segmented images.

All public functions are available from the package namespace::

    import tysserand as ty
    A = ty.build_delaunay(coords)

``from tysserand import tysserand as ty`` still works but is deprecated
(``FutureWarning``) and will be removed in a future release.

Submodules:

- ``build``: reconstruction from coordinates (Delaunay, knn, radial, lattice)
- ``contacting``: reconstruction from segmented images (contacting areas)
- ``convert``: CSR / edge pairs / DataFrame conversions, attribute matrices
- ``distances``: edge distances, trimming, weights
- ``io``: network persistence as ``.npz``
- ``plot``: network plotting
- ``interop``: NetworkX, iGraph and AnnData bridges
- ``annotation``: interactive visualization and annotation in napari
- ``scoring``: comparison with a ground truth network
- ``generators``: synthetic data

Heavy optional libraries (napari, OpenCV, scikit-image, scikit-learn,
pynndescent, matplotlib, seaborn...) are imported inside the functions that
use them, so that importing tysserand stays light, also in parallel workers.
"""

from .annotation import *
from .build import *
from .contacting import *
from .convert import *
from .distances import *
from .generators import *
from .interop import *
from .io import *
from .plot import *
from .scoring import *
