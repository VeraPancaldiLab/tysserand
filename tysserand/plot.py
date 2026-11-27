"""
Network plotting with matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Colormap
    from numpy.typing import ArrayLike

__all__ = [
    'rescale',
    'plot_network',
    'plot_network_distances',
    'showim',
]


def rescale(
    data: ArrayLike,
    perc_mini: float = 1,
    perc_maxi: float = 99,
    out_mini: float = 0,
    out_maxi: float = 1,
    cutoff_mini: bool = True,
    cutoff_maxi: bool = True,
    return_extrema: bool = False,
) -> np.ndarray | tuple[np.ndarray, float, float]:
    """
    Normalize the intensities of a planar 2D image.

    Parameters
    ----------
    data : numpy array
        the matrix to process
    perc_mini : float
        the low input level to set to the low output level
    perc_maxi : float
        the high input level to set to the high output level
    out_mini : int or float
        the low output level
    out_maxi : int or float
        the high output level
    cutoff_mini : bool
        if True sets final values below the low output level to the low output level
    cutoff_maxi : bool
        if True sets final values above the high output level to the high output level
    return_extrema : bool
        if True minimum and maximum percentiles of original data are also returned

    Returns
    -------
    data_out : numpy array
        the output image
    """

    mini = np.percentile(data, perc_mini)
    maxi = np.percentile(data, perc_maxi)
    if out_mini is None:
        out_mini = mini
    if out_maxi is None:
        out_maxi = maxi
    data_out = data - mini
    data_out = data_out * (out_maxi - out_mini) / (maxi - mini)
    data_out = data_out + out_mini
    if cutoff_mini:
        data_out[data_out < out_mini] = out_mini
    if cutoff_maxi:
        data_out[data_out > out_maxi] = out_maxi
    if return_extrema:
        return data_out, mini, maxi
    else:
        return data_out


def plot_network(
    coords: np.ndarray,
    A: csr_matrix,
    disp_id: bool = False,
    labels: ArrayLike | None = None,
    color_mapper: dict | None = None,
    show_legend: bool = True,
    legend_opt: dict | None = None,
    col_nodes: str | ArrayLike | None = None,
    cmap_nodes: str | Colormap | list | None = None,
    marker: str | None = None,
    size_nodes: float | ArrayLike | None = None,
    col_edges: str = 'k',
    alpha_edges: float = 0.5,
    linewidth: float | None = None,
    skip_edges: bool = False,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (15, 15),
    aspect: str = 'equal',
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a network.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...).
    A : scipy.sparse.csr_matrix
        CSR adjacency matrix.
    disp_id : bool, optional
        If True, nodes' indices are displayed. The default is False.
    labels : array-like or pd.Series, optional
        Node labels from which they are colored.
    color_mapper : dict, optional
        Maps each label to its color. Computed if not provided.
    show_legend : bool, optional
        If True, a legend is displayed. The default is True.
    legend_opt : dict or None, optional
        Optional parameters for the legend.
    col_nodes : str, optional
        Color of nodes. The default is None.
    cmap_nodes : list, optional
        List of hexadecimal colors for nodes attributes.
    marker : str, optional
        Marker used to display nodes. The default is None.
    size_nodes : int, optional
        Size of nodes. The default is None.
    col_edges : str, optional
        Color of edges. The default is 'k'.
    alpha_edges : float, optional
        Transparency of edges. The default is 0.5.
    linewidth : float, optional
        Width of edges. The default is None.
    skip_edges : bool, optional
        If True, edges are not displayed. The default is False.
    ax : matplotlib Axes, optional
        If provided, the plot is displayed in ax. The default is None.
    aspect : str, optional
        Control aspect ratio of the figure. The default is 'equal'.
    **kwargs : dict
        Optional parameters to display nodes.

    Returns
    -------
    None or (fig, ax) if ax was not provided.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.collections import LineCollection

    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax_none = False
    # plot nodes
    if labels is not None:
        if isinstance(labels, np.ndarray):
            uniq = np.unique(labels)
        elif isinstance(labels, pd.Series):
            uniq = labels.unique()
        else:
            uniq = np.unique(np.array(labels))
        if color_mapper is None:
            if cmap_nodes is None:
                cmap_nodes = sns.color_palette('muted').as_hex()
            n_colors = len(cmap_nodes)
            color_mapper = {x: cmap_nodes[i % n_colors] for i, x in enumerate(uniq)}
        for label in uniq:
            select = labels == label
            color = color_mapper[label]
            ax.scatter(
                coords[select, 0],
                coords[select, 1],
                c=color,
                label=label,
                marker=marker,
                s=size_nodes,
                zorder=10,
                **kwargs,
            )
        if show_legend:
            if legend_opt is None:
                ax.legend()
            else:
                ax.legend(**legend_opt)
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=col_nodes,
            cmap=cmap_nodes,
            marker=marker,
            s=size_nodes,
            zorder=10,
            **kwargs,
        )
    # plot edges using LineCollection
    if not skip_edges:
        rows, cols = A.nonzero()
        mask = rows < cols  # upper triangle only for undirected edges
        rows, cols = rows[mask], cols[mask]
        segments = np.stack([coords[rows, :2], coords[cols, :2]], axis=1)
        lc = LineCollection(
            segments,
            colors=col_edges,
            alpha=alpha_edges,
            linewidths=linewidth,
            zorder=5,
        )
        ax.add_collection(lc)
        ax.autoscale_view()
    if disp_id:
        offset = 0.02
        for i, (x, y) in enumerate(coords):
            ax.text(x - offset, y - offset, str(i), zorder=15)
    if aspect is not None:
        ax.set_aspect(aspect)
    if ax_none:
        return fig, ax


def plot_network_distances(
    coords: np.ndarray,
    D: csr_matrix,
    labels: ArrayLike | None = None,
    color_mapper: dict | None = None,
    show_legend: bool = True,
    legend_opt: dict | None = None,
    col_nodes: str | ArrayLike | None = None,
    cmap_nodes: str | Colormap | list | None = None,
    marker: str | None = None,
    size_nodes: float | ArrayLike | None = None,
    cmap_edges: str | Colormap = 'viridis',
    alpha_edges: float = 0.7,
    linewidth: float | None = None,
    figsize: tuple[float, float] = (15, 15),
    ax: plt.Axes | None = None,
    aspect: str = 'equal',
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a network with edges colored by their length.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...).
    D : scipy.sparse.csr_matrix
        CSR matrix of edge distances as returned by `distance_neighbors`.
    labels : array-like or pd.Series, optional
        Node labels from which they are colored.
    color_mapper : dict, optional
        Maps each label to its color. Computed if not provided.
    show_legend : bool, optional
        If True, a legend is displayed. The default is True.
    legend_opt : dict or None, optional
        Optional parameters for the legend.
    col_nodes : str, optional
        Color of nodes. The default is None.
    cmap_nodes : list, optional
        List of hexadecimal colors for nodes attributes.
    marker : str, optional
        Marker used to display nodes. The default is None.
    size_nodes : float, optional
        Size of nodes. The default is None.
    cmap_edges : str, optional
        Colormap of edges. The default is 'viridis'.
    alpha_edges : float, optional
        Transparency of edges. The default is 0.7.
    linewidth : float, optional
        Width of edges. The default is None.
    figsize : (float, float), optional
        Width, height in inches. The default is (15, 15).
    ax : matplotlib Axes, optional
        If provided, the plot is displayed in ax. The default is None.
    aspect : str, optional
        Proportions of the figure. The default is 'equal'.

    Returns
    -------
    None or (fig, ax) if ax was not provided.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.collections import LineCollection

    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax_none = False
        fig = ax.get_figure()
    # plot nodes
    if labels is not None:
        if isinstance(labels, np.ndarray):
            uniq = np.unique(labels)
        elif isinstance(labels, pd.Series):
            uniq = labels.unique()
        else:
            uniq = np.unique(np.array(labels))
        if color_mapper is None:
            if cmap_nodes is None:
                cmap_nodes = sns.color_palette('muted').as_hex()
            n_colors = len(cmap_nodes)
            color_mapper = {x: cmap_nodes[i % n_colors] for i, x in enumerate(uniq)}
        for label in uniq:
            select = labels == label
            color = color_mapper[label]
            ax.scatter(
                coords[select, 0],
                coords[select, 1],
                c=color,
                label=label,
                marker=marker,
                s=size_nodes,
                zorder=10,
                **kwargs,
            )
        if show_legend:
            if legend_opt is None:
                ax.legend()
            else:
                ax.legend(**legend_opt)
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=col_nodes,
            cmap=cmap_nodes,
            marker=marker,
            s=size_nodes,
            zorder=10,
            **kwargs,
        )
    # plot edges
    rows, cols = D.nonzero()
    # keep only upper triangle to draw each undirected edge once
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    distances = np.array(D[rows, cols]).ravel()

    segments = np.stack([coords[rows, :2], coords[cols, :2]], axis=1)
    cmap = mpl.cm.get_cmap(cmap_edges)
    norm = mpl.colors.Normalize(vmin=distances.min(), vmax=distances.max())
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        alpha=alpha_edges,
        linewidths=linewidth,
        zorder=0,
    )
    lc.set_array(distances)
    ax.add_collection(lc)
    ax.autoscale_view()
    fig.colorbar(lc, ax=ax, orientation='vertical', label='Distance')

    if aspect is not None:
        ax.set_aspect(aspect)
    if ax_none:
        return fig, ax


def showim(
    image: np.ndarray, figsize: tuple[float, float] = (9, 9), ax: plt.Axes | None = None, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """
    Displays an image with thigh layout and without axes.

    Parameters
    ----------
    image : ndarray
        A 1 or 3 channels images.
    figsize : (int, int), optional
        Size of the figure. The default is (9,9).
    ax : matplotlib ax object, optional
        If provided, the plot is displayed in ax. The default is None.
    **kwargs : dic
        Other options for plt.imshow().

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        return_ax = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        return_ax = False
    ax.imshow(image, **kwargs)
    ax.axis('off')
    ax.figure.tight_layout()
    if return_ax:
        return fig, ax
