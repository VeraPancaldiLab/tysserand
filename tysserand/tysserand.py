import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import libpysal
from libpysal.cg.voronoi  import voronoi, voronoi_frames
from libpysal.weights import Queen, Rook
import geopandas

from scipy.spatial import Voronoi
from sklearn.neighbors import BallTree

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

def pairs_from_NN(ind):
    """
    Convert a matrix of Neirest Neighbors indices into 
    a matrix of unique pairs of neighbors

    Parameters
    ----------
    ind : ndarray
        The (n_objects x n_neighbors) matrix of neighbors indices.

    Returns
    -------
    pairs : ndarray
        The (n_pairs x 2) matrix of neighbors indices.

    """
    
    NN = ind.shape[1]
    source_nodes = np.repeat(ind[:,0], NN-1).reshape(-1,1)
    target_nodes = ind[:,1:].reshape(-1,1)
    pairs = np.hstack((source_nodes, target_nodes))
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs

def build_NN(coords, k=6, **kwargs):
    tree = BallTree(coords, **kwargs)
    _, ind = tree.query(coords, k=k)
    pairs = pairs_from_NN(ind)
    return pairs

def build_within_radius(coords, r, **kwargs):
    tree = BallTree(coords, **kwargs)
    ind = tree.query_radius(coords, r=r)
    # clean arrays of neighbors from self referencing neighbors
    # and aggregate at the same time
    source_nodes = []
    target_nodes = []
    for i, arr in enumerate(ind):
        neigh = arr[arr != i]
        source_nodes.append([i]*(neigh.size))
        target_nodes.append(neigh)
    # flatten arrays of arrays
    import itertools
    source_nodes = np.fromiter(itertools.chain.from_iterable(source_nodes), int).reshape(-1,1)
    target_nodes = np.fromiter(itertools.chain.from_iterable(target_nodes), int).reshape(-1,1)
    # remove duplicate pairs
    pairs = np.hstack((source_nodes, target_nodes))
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs

def plot_network(coords, pairs, figsize=(15, 15), col_nodes=None, marker=None,
                 size_nodes=None, col_edges='k', alpha_edges=0.5, ax=None, 
                 aspect='equal', **kwargs):
    """
    **kwargs: labels of nodes
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # plot nodes
    ax.scatter(coords[:,0], coords[:,1], c=col_nodes, marker=marker, s=size_nodes, zorder=10, **kwargs)
    # plot edges
    for pair in pairs[:,:]:
        [x0, y0], [x1, y1] = coords[pair]
        ax.plot([x0, x1], [y0, y1], c=col_edges, zorder=5, alpha=alpha_edges)
    if aspect is not None:
        ax.set_aspect(aspect)

def rescale(data, perc_mini=1, perc_maxi=99, 
            out_mini=0, out_maxi=1, 
            cutoff_mini=True, cutoff_maxi=True, 
            return_extrema=False):
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
    data_out = data_out * (out_maxi-out_mini) / (maxi-mini)
    data_out = data_out + out_mini
    if cutoff_mini:
        data_out[data_out<out_mini] = out_mini
    if cutoff_maxi:
        data_out[data_out>out_maxi] = out_maxi
    if return_extrema:
        return data_out, mini, maxi
    else:
        return data_out

def plot_network_distances(coords, pairs, distances,  
                           col_nodes=None, marker=None, size_nodes=None, 
                           cmap_edges='viridis', alpha_edges=0.7, 
                           figsize=(15, 15), ax=None, aspect='equal', **kwargs):
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
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # plot nodes
    ax.scatter(coords[:,0], coords[:,1], c=col_nodes, marker=marker, s=size_nodes, zorder=10, **kwargs)
    # plot edges
    scaled_dist, min_dist, max_dist = rescale(distances, return_extrema=True)
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
    for pair, dist in zip(pairs[:,:], scaled_dist):
        [x0, y0], [x1, y1] = coords[pair]
        ax.plot([x0, x1], [y0, y1], c=cmap(dist), zorder=0, alpha=alpha_edges)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='vertical', label='Distance')
    # TODO: plot many lines more efficiently check
    # from https://stackoverflow.com/a/50029441
    # https://matplotlib.org/gallery/shapes_and_collections/line_collection.html#sphx-glr-gallery-shapes-and-collections-line-collection-py
    
    if aspect is not None:
        ax.set_aspect(aspect)

def showim(image, figsize=(9,9), ax=None, **kwargs):
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

###### TODO ######

# methods to construct networks:
#   - contact (segmented images, ...)
#   - knn
#   - distance sphere
#   - TDA based?
#   - Voronoi: OK
#   - Gabriel

# plots to find distance threshold:
#     - static: OK
#     - interactive with cursor for maw distance and color pops when edge is discarded
# plots to find appropriate number of neighbors

# Examples and benchmarks with:
#     - very small network
#     - tile from WSI
#     - WSI data 
#       (/home/alexis/Projects/Image_to_Network/data/raw/WSI/conversion/converted)
