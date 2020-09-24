import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import libpysal
from libpysal.cg.voronoi  import voronoi, voronoi_frames
from libpysal.weights import Queen, Rook
import geopandas

from scipy.spatial import Voronoi

def make_simple_coords():
    """
    Makes really simple coordinates to illustrate network construction methods.

    Returns
    -------
    coords : ndarray
        Array with 1st and 2nd column corresponding to x and y coordinates.
    """
    
    x = np.array([144, 124, 97, 165, 114, 60, 165, 0, 76, 50, 147])
    y = np.array([ 0, 3, 21, 28, 34, 38, 51, 54, 58, 56, 61])
    coords = np.vstack((x,y)).T
    return coords

def distance_neighbors(coords, pairs):
    """
    Compute all distances between neighbors in a network.
    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    pairs : ndarray
        Pairs of neighbors given by the first and second element of each row.

    Returns
    -------
    distances : array
        Distances between each pair of neighbors.

    """
    
    # positions of pairs of neighbors as x0, y0, x1, y1
    pos = coords[pairs]
    pos = np.zeros((pairs.shape[0],4))
    pos[:,[0,1]] = coords[pairs[:,0], :]
    pos[:,[2,3]] = coords[pairs[:,1], :]
    distances = np.sqrt((pos[:,0]-pos[:,2])**2+(pos[:,1]-pos[:,3])**2)
    return distances

def find_trim_dist(dist, method, param):
    if method == 'percentile':
        dist_thresh = np.percentile(dist, param)
    return dist_thresh

def build_voronoi(coords, trim_dist='percentile', trim_param=0.5, return_dist=False):
    """
    Examples
    --------
    >>> x = np.array([144, 124,  97, 165, 114,  60, 165,   0,  76,  50, 147])
    >>> y = np.array([ 0,  3, 21, 28, 34, 38, 51, 54, 58, 56, 61])
    >>> coords = np.vstack((x,y)).T
    >>> tesselation(coords)


    >>> coords = pd.DataFrame({'x':x, 'y':y})
    >>> tessellation(coords[['x','y']])
    """

    # pairs of indices of neighbors
    pairs = Voronoi(coords).ridge_points

    if trim_dist is not False:
        dist = distance_neighbors(coords, pairs)
        if not isinstance(trim_dist, (int, float)):
            trim_dist = find_trim_dist(dist=dist, method=trim_dist, param=trim_param)
        pairs = pairs[dist < trim_dist, :]
    return pairs

# from https://stackoverflow.com/a/50029441
from matplotlib.collections import LineCollection
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def plot_network(coords, pairs, figsize=(15, 15), col_nodes=None, marker=None,
                 size_nodes=None, col_edges='k', alpha_edges=0.5, aspect=None, **kwargs):
    """
    **kwargs: labels of nodes
    """
    fig, ax = plt.subplots(figsize=figsize)
    # plot nodes
    ax.scatter(coords[:,0], coords[:,1], c=col_nodes, marker=marker, s=size_nodes, zorder=10, **kwargs)
    # plot edges
    for pair in pairs[:,:]:
        [x0, y0], [x1, y1] = coords[pair]
        ax.plot([x0, x1], [y0, y1], c=col_edges, zorder=0, alpha=alpha_edges)
    plt.legend()
    if aspect is not None:
        ax.set_aspect(aspect)

# def plot_network_big(coords, pairs, figsize=(15, 15), col_nodes=None, marker=None,
#                  size_nodes=None, col_edges='k', alpha_edges=0.5, aspect=None, **kwargs):
#     """
#     **kwargs: labels of nodes
#     """
#     fig, ax = plt.subplots(figsize=figsize)
#     # plot nodes
#     ax.scatter(coords[:,0], coords[:,1], c=col_nodes, marker=marker, s=size_nodes, zorder=10, **kwargs)
#     # plot edges
#     # obtain 3D array of ([[x0, y0], [x1, y1]])
#     coord_pairs = coords[pairs]
    
#     multiline(coords[:,0], coords[:,0], c=np.zeros(pairs.shape[0]), ax=ax, 
#               zorder=0, alpha=alpha_edges, **kwargs)
#     for pair in pairs[:,:]:
#         [x0, y0], [x1, y1] = coords[pair]
#         ax.plot([x0, x1], [y0, y1], c=col_edges, zorder=0, alpha=alpha_edges)
#     plt.legend()
#     if aspect is not None:
#         ax.set_aspect(aspect)


# check this out too
# https://matplotlib.org/gallery/shapes_and_collections/line_collection.html#sphx-glr-gallery-shapes-and-collections-line-collection-py

def plot_network_distances(coords, pairs, distances,  
                           col_nodes=None, marker=None, size_nodes=None, 
                           cmap_edges='viridis', alpha_edges=0.7, 
                           figsize=(15, 15), aspect=None, **kwargs):
    """
    Plot a network with edges colored by their length.

    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    pairs : ndarray
        Pairs of neighbors given by the first and second element of each row.
    distances : array
        Distances between each pair of neighbors.
    col_nodes : TYPE, optional
        DESCRIPTION. The default is None.
    marker : str, optional
        Marker for nodes. The default is None.
    size_nodes : float, optional
        Size of nodes. The default is None.
    cmap_edges : str of matplotlib.colormap, optional
        Colormap of edges. The default is 'viridis'.
    alpha_edges : float, optional
        Tansparency of edges. The default is 0.7.
    figsize : (float, float), default: :rc:`figure.figsize`
        Width, height in inches. The default is (15, 15).
    aspect : str, optional
        Proportions of the figure. The default is None.
    **kwargs : TYPE
        labels of nodes.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=figsize)
    # plot nodes
    ax.scatter(coords[:,0], coords[:,1], c=col_nodes, marker=marker, s=size_nodes, zorder=10, **kwargs)
    # plot edges
    scaled_dist = distances / distances.max()
    for pair, dist in zip(pairs[:,:], scaled_dist):
        [x0, y0], [x1, y1] = coords[pair]
        ax.plot([x0, x1], [y0, y1], c=plt.cm.viridis(dist), zorder=0, alpha=alpha_edges)
    plt.colorbar()
    if aspect is not None:
        ax.set_aspect(aspect)

###### TODO ######

# methods to construct networks:
#   - contact (segmented images, ...)
#   - knn
#   - distance sphere
#   - TDA based?
#   - Voronoi
#   - Gabriel

# plots to find distance threshold:
#     - static
#     - interactive with cursor for maw distance and color pops when edge is discarded

# Examples and benchmarks with:
#     - very small network
#     - tile from WSI
#     - WSI data 
#       (/home/alexis/Projects/Image_to_Network/data/raw/WSI/conversion/converted)
