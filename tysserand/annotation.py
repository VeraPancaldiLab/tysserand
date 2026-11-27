"""
Interactive visualization and annotation of networks in napari.
"""

from __future__ import annotations
import numpy as np
import joblib

from .convert import csr_to_edge_coords, vectors_to_csr

from typing import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pathlib import Path
    from scipy.sparse import csr_matrix
    import napari
    import pandas as pd

__all__ = [
    'visualize',
    'get_annotation_names',
    'convert_nodes_tys_to_nap',
    'convert_edges_tys_to_nap',
    'make_annotation_dict',
    'get_annotation_dict',
    'save_annotations',
    'load_annotations',
    'add_nodes',
    'add_edges',
    'add_annotations',
    'assign_nodes_to_edges',
    'update_edges',
    'update_annotations',
]


def visualize(viewer: napari.Viewer, img: np.ndarray, colormaps: dict | None = None) -> None:
    """
    Create a napari viewer instance with image splitted into
    separate channels.
    """
    
    if colormaps == 'rgb':
        colormaps = [
            'red',
            'green',
            'blue',
        ]
    # add successively all channels
    for i in range(img.shape[-1]):
        # avoid the alpha channel of RGB images
        if i == 3 and np.all(img[:, :, i] == 1):
            pass
        else:
            if colormaps is not None and i < len(colormaps):
                colormap = colormaps[i]
            else:
                colormap = 'gray'
            viewer.add_image(img[:, :, i], name='ch' + str(i), colormap=colormap, blending='additive')
    return


def get_annotation_names(viewer: napari.Viewer) -> tuple[str, str]:
    """Detect the names of nodes and edges layers"""
    import napari

    layer_nodes_name = None
    layer_edges_name = None
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.points.points.Points):
            layer_nodes_name = layer.name
        elif isinstance(layer, napari.layers.shapes.shapes.Shapes):
            layer_edges_name = layer.name
        if layer_nodes_name is not None and layer_edges_name is not None:
            break
    return layer_nodes_name, layer_edges_name


def convert_nodes_tys_to_nap(coords: np.ndarray) -> np.ndarray:
    """
    Reverse the x, y, z ordering of nodes coordinates to match napari's convention.
    """
    new_nodes = coords[:, ::-1]
    return new_nodes


def convert_edges_tys_to_nap(coords: np.ndarray, pairs: np.ndarray) -> list[np.ndarray]:
    """
    Make start and end coordinates of edges for napari.

    .. deprecated::
        Use :func:`csr_to_edge_coords` instead, which accepts a CSR adjacency matrix directly.
    """
    import warnings
    warnings.warn(
        "convert_edges_tys_to_nap is deprecated; use csr_to_edge_coords(coords, A) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    new_edges = []
    for pair in pairs[:,:]:
        new_edges.append(np.array(coords[pair]))
    return new_edges


def make_annotation_dict(coords: np.ndarray, A: csr_matrix | None = None, pairs: np.ndarray | None = None,
                         nodes_class: ArrayLike | None = None,
                         nodes_class_color_mapper: dict | None = None,
                         ) -> dict:
    """
    Create a dictionary of annotations from tysserand network objects.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates in tysserand convention (x, y, ...), shape (n_nodes, ndim).
    A : scipy.sparse.csr_matrix, optional
        CSR adjacency matrix. Preferred over `pairs`.
    pairs : np.ndarray, optional
        Deprecated. Edge list of shape (n_edges, 2). Use `A` instead.
    nodes_class : array-like, optional
        Class label per node.
    nodes_class_color_mapper : dict, optional
        Maps each class label to a color.

    Returns
    -------
    annotations : dict
    """
    import warnings
    if pairs is not None:
        warnings.warn(
            "The `pairs` argument to make_annotation_dict is deprecated; "
            "pass A (csr_matrix) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if A is None:
            A = vectors_to_csr(pairs, n_nodes=len(coords))

    annotations = {}
    new_nodes = convert_nodes_tys_to_nap(coords)
    annotations['nodes_coords'] = new_nodes
    if nodes_class is not None:
        annotations['nodes_class'] = nodes_class
    if nodes_class_color_mapper is not None:
        annotations['nodes_class_color_mapper'] = nodes_class_color_mapper
    if A is not None:
        annotations['edges_coords'] = csr_to_edge_coords(new_nodes, A)
        annotations['A'] = A
    return annotations


def get_annotation_dict(viewer: napari.Viewer, layer_nodes_name: str, layer_edges_name: str) -> dict:
    """
    Create a dictionnary of annotations from layers in a napari viewer.
    """

    annotations = {}
    if layer_nodes_name is not None:
        annotations['nodes_coords'] = viewer.layers[layer_nodes_name].data
        # pick a unique value instead of saving a 2D array of duplicates
        annotations['nodes_size'] = np.median(viewer.layers[layer_nodes_name].size)
        
        # ------ convert colors arrays into unique nodes classes ------
        colors = viewer.layers[layer_nodes_name].face_color
        color_set = {tuple(e) for e in colors}
        # mapper to convert nodes classes into color tuples
        nodes_class_color_mapper = dict(zip(range(len(color_set)), color_set))
        # mapper to convert color tuples into nodes classes
        nodes_color_class_mapper = {val: key for key, val in nodes_class_color_mapper.items()}
        nodes_class = np.array([nodes_color_class_mapper[tuple(key)] for key in colors])
        
        annotations['nodes_class'] = nodes_class
        annotations['nodes_class_color_mapper'] = nodes_class_color_mapper
        
    if layer_edges_name is not None:
        annotations['edges_coords'] = viewer.layers[layer_edges_name].data
        annotations['edges_edge_width'] = np.median(viewer.layers[layer_edges_name].edge_width)
        # TODO (maybe): implement edge color mapper
        # annotations['edges_edge_colors'] = viewer.layers[layer_edges_name].edge_color
    return annotations


def save_annotations(path: str | Path, viewer: napari.Viewer | None = None, annotations: dict | None = None, layer_names: Sequence[str] | None = None) -> None:
    """"
    Create and save annotations in the layers of a napari viewer.
    """
    if annotations is None:
        if layer_names is not None:
            layer_nodes_name, layer_edges_name = layer_names
        else:
            layer_nodes_name, layer_edges_name = get_annotation_names(viewer)
        annotations = get_annotation_dict(viewer, layer_nodes_name, layer_edges_name)
    joblib.dump(annotations, path);
    return


def load_annotations(path: str | Path) -> dict:
    """"
    Load annotations for the layers of a napari viewer.
    """
    annotations = joblib.load(path);
    return annotations


def add_nodes(
    viewer: napari.Viewer, 
    annotations: dict,
    name: str = 'nodes',
    ) -> None:
    """
    Add nodes annotations in a napari viewer.
    """
    if 'nodes_class_color_mapper' in annotations.keys() and 'nodes_class' in annotations.keys():
        face_color = np.array([annotations['nodes_class_color_mapper'][key] for key in annotations['nodes_class']])
    else:
        face_color = '#1f77b4'
    if 'nodes_size' in annotations.keys():
        size = annotations['nodes_size']
    else:
        size = 10
    viewer.add_points(
        annotations['nodes_coords'],
        # reconstruct the colors array
        face_color=face_color,
        size=size,
        name=name,
        )
    return


def add_edges(
    viewer: napari.Viewer, 
    annotations: dict,
    edge_color: str = 'white',
    name: str = 'edges',
    ) -> None:
    """
    Add edges annotations in a napari viewer.
    """
    
    if 'edge_width' in annotations.keys():
        edge_width = annotations['edge_width']
    else: 
        edge_width = 1
    viewer.add_shapes(
        annotations['edges_coords'], 
        shape_type='line', 
        edge_width=edge_width,
        edge_color=edge_color,
        name=name,
    )
    return


def add_annotations(
    viewer: napari.Viewer,
    annotations: dict,
    layer_nodes_name: str = 'nodes',
    layer_edges_name: str = 'edges',
    edge_color: str = 'white',
    ) -> None:
    """
    Add nodes and edges annotations in a napari viewer.
    """
    
    if 'nodes_coords' in annotations.keys():
        add_nodes(viewer, annotations, name=layer_nodes_name)
    if 'edges_coords' in annotations.keys():
        add_edges(viewer, annotations, edge_color=edge_color, name=layer_edges_name)
    return


def assign_nodes_to_edges(nodes: pd.DataFrame, edges: np.ndarray, n_nodes: int | None = None) -> tuple[np.ndarray, csr_matrix]:
    """
    Snap edge endpoints to nearest nodes and return a CSR adjacency matrix.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates in napari convention, shape (n_nodes, ndim).
    edges : list of np.ndarray
        Napari Shapes edge data: list of (2, ndim) arrays, one per edge.
    n_nodes : int, optional
        Total number of nodes. If None, ``len(nodes)`` is used. Pass explicitly
        when isolated nodes are present to ensure the CSR matrix has the correct size.

    Returns
    -------
    new_edges : list of np.ndarray
        Snapped edge coordinates in napari format (same structure as `edges`).
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix recovered from the snapped edges.
    """
    from scipy.spatial import cKDTree

    if n_nodes is None:
        n_nodes = len(nodes)

    edges_arr = np.vstack(edges)
    kdt_nodes = cKDTree(nodes)

    # closest node id; discard computed distances
    _, pairs = kdt_nodes.query(x=edges_arr, k=1)
    # reshape flat list [src0, tgt0, src1, tgt1, ...] into (n_edges, 2)
    pairs = np.vstack((pairs[::2], pairs[1::2])).T

    new_edges = [np.array(nodes[pair]) for pair in pairs]
    A = vectors_to_csr(pairs, n_nodes=n_nodes)
    return new_edges, A


def update_edges(
    viewer: napari.Viewer, 
    annotations: dict, 
    edge_color: str = 'white',
    name: str = 'edges',
    ) -> None:    
    """
    Replace edges annotations with new ones in a napari viewer.
    """
    
    try:
        del viewer.layers[name]
    except ValueError:
        # edges layer was already deleted
        pass
    add_edges(viewer, annotations, edge_color=edge_color, name=name)


def update_annotations(
    viewer: napari.Viewer,
    annotations: dict,
    layer_nodes_name: str = 'nodes',
    layer_edges_name: str = 'edges',
    edge_color: str = 'white',
    ) -> None:
    """
    Update nodes and edges annotations in a napari viewer.
    """
    
    try:
        del viewer.layers[layer_nodes_name]
    except ValueError:
        # nodes layer was already deleted
        pass
    try:
        del viewer.layers[layer_edges_name]
    except ValueError:
        # edges layer was already deleted
        pass

    if 'nodes_coords' in annotations.keys():
        add_nodes(viewer, annotations, name=layer_nodes_name)
    if 'edges_coords' in annotations.keys():
        add_edges(viewer, annotations, edge_color=edge_color, name=layer_edges_name)
    return
