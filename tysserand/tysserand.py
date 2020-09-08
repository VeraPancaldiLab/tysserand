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

def distance_neighbors(coords, pairs):
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

def tessellation(coords, trim_dist = 'percentile', trim_param = 0.5, return_dist = False):
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

def plot_network(coords, pairs, figsize=(15, 15), col_nodes=None, marker=None,
                 size_nodes=None, col_edges = 'k', alpha_edges = 0.5, aspect=None, **kwargs):
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
