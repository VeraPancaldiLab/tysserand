import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import itertools
import libpysal
from libpysal.cg.voronoi  import voronoi, voronoi_frames
from libpysal.weights import Queen, Rook
import geopandas

from scipy.spatial import Voronoi
from sklearn.neighbors import BallTree
from skimage import morphology, feature, measure, segmentation, filters, color
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix

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

def make_random_nodes(size=100, ndim=2, expand=True):
    """
    Make a random set of nodes

    Parameters
    ----------
    size : int, optional
        Number of nodes. The default is 100.
    ndim : int, optional
        Number of dimensions. The default is 2.
    expand : bool, optional
        If True, positions are multiplied by size**(1/ndim) in order to have a 
        consistent spacing across various `size` and `ndim` values. 
        The default is True.

    Returns
    -------
    coords : ndarray
        Coordinates of the set of nodes.
    """
    
    coords = np.random.random(size=size*ndim).reshape((-1,ndim))
    if expand:
        coords = coords * size**(1/ndim)
    return coords

def make_random_tiles(sx=500, sy=500, nb=50, noise_sigma=None,
                      regular=True, double_pattern=False, 
                      assym_y=True, return_image=False):
    """
    Build contacting areas similar to cell segmentation in tissues.

    Parameters
    ----------
    sx : int, optional
        Size of the image on the x axis. The default is 500.
    sy : int, optional
        Size of the image on the x axis. The default is 500.
    nb : int, optional
        Related to the number of points, but not equal. The default is 50.
    noise_sigma : None or float, optional
        If float, a gaussian noise is added to seeds positions.
    regular : bool, optional
        If True points are on a regular lattice, else they are randomly located. 
        The default is True.
    double_pattern : bool, optional
        If True the regular lattice has more points. The default is False.
    assym_y : bool, optional
        If True the frenquency of seeds is twice higher on the y-axis. The default is True.
    return_image : bool, optional
        If True the image of seed points is also returned. The default is False.

    Returns
    -------
    coords : ndarray
        Coordinates of the set of nodes.
    masks : ndarray
        Detected areas coded by a unique integer.
        
    Examples
    --------
    >>> coords, masks, image = make_random_tiles(double_pattern=True, return_image=True)
    >>> showim(image)
    >>> label_cmap = mpl.cm.get_cmap('Set2')(range(8))
    >>> showim(color.label2rgb(masks, bg_label=0, colors=label_cmap), origin='lower')

    """
    
    image = np.zeros((sy, sx))
    if regular:
        x = np.linspace(start=0, stop=sx-1, num=nb, dtype=int)
        x = np.hstack((x[::2], x[1::2]))
        if assym_y:
            nb = nb*2
        y = np.linspace(start=0, stop=sy-1, num=nb, dtype=int)
        if double_pattern:
            y = np.hstack((y[::2], y[1::2]))
        x_id = np.tile(x, y.size//2)
        y_id = np.repeat(y, x.size//2)
    else:
        x_id = np.random.randint(sx, size=nb)
        y_id = np.random.randint(sy, size=nb)
        
    if noise_sigma is not None:
        x_id = x_id + np.random.normal(loc=0.0, scale=noise_sigma, size=x_id.size)
        x_id[x_id<0] = 0
        x_id[x_id>sx-1] = sx-1
        x_id = np.round(x_id).astype(int)
        y_id = y_id + np.random.normal(loc=0.0, scale=noise_sigma, size=y_id.size)
        y_id[y_id<0] = 0
        y_id[y_id>sy-1] = sy-1
        y_id = np.round(y_id).astype(int)
        
    coords = np.vstack((x_id, y_id)).T
    image[y_id, x_id] = 1
    masks = segmentation.watershed(-image)
    
    if return_image:
        return coords, masks, image
    else:
        return coords, masks


def distance_neighbors(coords, pairs):
    """
    Compute all distances between neighbors in a network.
    
    Parameters
    ----------
    coords : dataframe
        Coordinates of points where columns are 'x', 'y', ...
    pairs : ndarray
        The (n_pairs x 2) array of neighbors indices.

    Returns
    -------
    distances : array
        Distances between each pair of neighbors.
    """
    
    # source nodes coordinates
    c0 = coords[pairs[:,0]]
    # target nodes coordinates
    c1 = coords[pairs[:,1]]
    distances = (c0 - c1)**2
    distances = np.sqrt(distances.sum(axis=1))
    return distances

def find_trim_dist(dist, method='percentile', param=99):
    """
    Find the distance threshold to eliminate reconstructed edges in a network.

    Parameters
    ----------
    dist : array
        Distances between pairs of nodes.
    method : str, optional
        Method used to compute the threshold. The default is 'percentile'.
    param : int or float, optional
        For the 'percentile' method it is the percentile of distances used as 
        the threshold. The default is 99.

    Returns
    -------
    dist_thresh : float
        Threshold distance.
    """
    if method == 'percentile':
        dist_thresh = np.percentile(dist, param)
    return dist_thresh

def build_delaunay(coords, trim_dist='percentile', trim_param=99, return_dist=False):
    """
    Reconstruct edges between nodes by Delaunay triangulation.

    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    trim_dist : str or float, optional
        Method or distance used to delete reconstructed edges. The default is 'percentile'.
    param : int or float, optional
        For the 'percentile' method it is the percentile of distances used as 
        the threshold. The default is 99.
    return_dist : bool, optional
        Whether distances are returned, usefull to try sevral trimming methods and parameters. 
        The default is False.
    
    Examples
    --------
    >>> coords = make_simple_coords()
    >>> pairs = build_delaunay(coords, trim_dist=False)

    Returns
    -------
    pairs : ndarray
        The (n_pairs x 2) array of neighbors indices.
    """

    # pairs of indices of neighbors
    pairs = Voronoi(coords).ridge_points

    if trim_dist is not False:
        dist = distance_neighbors(coords, pairs)
        if not isinstance(trim_dist, (int, float)):
            trim_dist = find_trim_dist(dist=dist, method=trim_dist, param=trim_param)
        pairs = pairs[dist < trim_dist, :]
    return pairs

def pairs_from_knn(ind):
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

def build_knn(coords, k=6, **kwargs):
    """
    Reconstruct edges between nodes by k-nearest neighbors (knn) method.
    An edge is drawn between each node and its k nearest neighbors.

    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    k : int, optional
        Number of nearest neighbors. The default is 6.
    
    Examples
    --------
    >>> coords = make_simple_coords()
    >>> pairs = build_knn(coords)

    Returns
    -------
    pairs : ndarray
        The (n_pairs x 2) matrix of neighbors indices.
    """
    
    tree = BallTree(coords, **kwargs)
    _, ind = tree.query(coords, k=k)
    pairs = pairs_from_knn(ind)
    return pairs

def build_rdn(coords, r, **kwargs):
    """
    Reconstruct edges between nodes by radial distance neighbors (rdn) method.
    An edge is drawn between each node and the nodes closer 
    than a threshold distance (within a radius).

    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    r : float, optional
        Radius in which nodes are connected.
    
    Examples
    --------
    >>> coords = make_simple_coords()
    >>> pairs = build_rdn(coords, r=60)

    Returns
    -------
    pairs : ndarray
        The (n_pairs x 2) matrix of neighbors indices.
    """
    
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
    source_nodes = np.fromiter(itertools.chain.from_iterable(source_nodes), int).reshape(-1,1)
    target_nodes = np.fromiter(itertools.chain.from_iterable(target_nodes), int).reshape(-1,1)
    # remove duplicate pairs
    pairs = np.hstack((source_nodes, target_nodes))
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs

def hyperdiagonal(coords):
    """
    Compute the maximum possible distance from a set of coordinates as the
    diagonal of the (multidimensional) cube they occupy.

    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)

    Returns
    -------
    dist : float
        Maximum possible distance.
    """
    
    mini = coords.min(axis=0)
    maxi = coords.max(axis=0)
    dist = (maxi - mini)**2
    dist = np.sqrt(dist.sum())
    return dist
    

def find_neighbors(masks, i, r=1):
    """
    Find the neighbors of a given mask.
    
    Parameters
    ----------
    masks : array_like
        2D array of integers defining the identity of masks
        0 is background (no object detected)
    i : int
        The mask for which we look for the neighbors.
    r : int
        Radius of search.
        
    Returns
    -------
    pairs : ndarray
        Pairs of neighbors given by the first and second element of each row, 
        values correspond to values in masks, which are different from index
        values of nodes
    """
    
    mask = masks == i
    # create the border in which we'll look at other masks
    if r == 1:
        selem = morphology.square(3)
    else:
        selem = morphology.disk(r)
    dilated = morphology.dilation(mask, selem)
    border = np.logical_xor(mask, dilated)
    # detect potential touching masks
    neighbors = np.unique(masks[border])
    # if there is only background
#     if neighbors.size == 1:
#         return False
#     else:
#         return neighbors[neighbors != 0]
    return neighbors[neighbors != 0]

def build_contacting(masks, r=1):
    """
    Build a network from segmented regions that contact each other or are 
    within a given distance from each other.

    Parameters
    ----------
    masks : array_like
        2D array of integers defining the identity of masks
        0 is background (no object detected)
    r : int
        Radius of search.

    Returns
    -------
    pairs : ndarray
        Pairs of neighbors given by the first and second element of each row, 
        values correspond to values in masks, which are different from index
        values of nodes

    """
    source_nodes = []
    target_nodes = []
    for i in range(1, masks.max()+1):
        neigh = find_neighbors(masks, i, r=r)
        source_nodes.append([i]*(neigh.size))
        target_nodes.append(neigh)
    # flatten arrays of arrays
    source_nodes = np.fromiter(itertools.chain.from_iterable(source_nodes), int).reshape(-1,1)
    target_nodes = np.fromiter(itertools.chain.from_iterable(target_nodes), int).reshape(-1,1)
    # remove duplicate pairs
    pairs = np.hstack((source_nodes, target_nodes))
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs

def mask_val_coord(masks):
    """
    Compute the mapping between mask regions and their centroid coordinates.

    Parameters
    ----------
    masks : array_like
        2D array of integers defining the identity of masks
        0 is background (no object detected)

    Returns
    -------
    coords : dataframe
        Coordinates of points with columns corresponding to axes ('x', 'y', ...)
    """
    
    coords = measure.regionprops_table(masks, properties=('label', 'centroid'))
    coords = pd.DataFrame.from_dict(coords)
    coords.rename(columns={'centroid-1':'x',  'centroid-0':'y'}, inplace=True)
    coords.index = coords['label']
    coords.drop(columns='label', inplace=True)
    return coords

def refactor_coords_pairs(coords, pairs):
    """
    Transforms coordinates and pairs of nodes data from segmented areas into
    the formats used by the other functions for network analysis and visualization.

    Parameters
    ----------
    coords : dataframe
        Coordinates of points with columns corresponding to axes ('x', 'y', ...)
    pairs : ndarray
        Pairs of neighbors given by the first and second element of each row, 
        values correspond to values in masks, which are different from index
        values of nodes

    Returns
    -------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    pairs : ndarray
        Pairs of neighbors given by the first and second element of each row.
    """
    
    coords = coords.loc[:, ['x', 'y']].values
    pairs = pairs - 1
    return coords, pairs

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

def plot_network(coords, pairs, disp_id=False, labels=None,
                 color_mapper=None, legend=True,
                 col_nodes=None, cmap_nodes=None, marker=None,
                 size_nodes=None, col_edges='k', alpha_edges=0.5, 
                 linewidth=None,
                 ax=None, figsize=(15, 15), aspect='equal', **kwargs):
    """
    Plot a network.

    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    pairs : ndarray
        The (n_pairs x 2) array of neighbors indices.
    disp_id: bool
        If True nodes' indices are displayed.
    labels: panda series
        The nodes' labels from which they are colored.
    color_mapper: dict
        Maps each label to its color. Computed if not provided.
    figsize : (float, float), default: :rc:`figure.figsize`
        Width, height in inches. The default is (15, 15).
    col_nodes : str of matplotlib compatible color, optional
        Color of nodes. The default is None.
    cmap_nodes: list
        List of hexadecimal colors for nodes attributes.
    marker : str, optional
        Marker used to display nodes. The default is None.
    size_nodes : int, optional
        Size of nodes. The default is None.
    col_edges : str or matplotlib compatible color, optional
        Color of edges. The default is 'k'.
    alpha_edges : float, optional
        Tansparency of edges. The default is 0.5.
    linewidth : float, optional
        Width of edges. The default is None.
    ax : matplotlib ax object, optional
        If provided, the plot is displayed in ax. The default is None.
    aspect : str, optional
        Control aspect ration of the figure. The default is 'equal'.
    **kwargs : dict
        Optional parameters to display nodes.

    Returns
    -------
    None or (fig, ax) if not provided in parameters.
    """
    
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
        # color nodes with manual colors
        if color_mapper is None:
            if cmap_nodes is None:
                att_colors = sns.color_palette('muted').as_hex() 
            color_mapper = dict(zip(uniq, att_colors))
        for label in uniq:
            select = labels == label
            color = color_mapper[label]
            ax.scatter(coords[select,0], coords[select,1], c=color, label=label,
                       marker=marker, s=size_nodes, zorder=10, **kwargs)
        if legend:
            plt.legend()
    else:
        ax.scatter(coords[:,0], coords[:,1], c=col_nodes, cmap=cmap_nodes, 
                   marker=marker, s=size_nodes, zorder=10, **kwargs)
    # plot edges
    for pair in pairs[:,:]:
        [x0, y0], [x1, y1] = coords[pair]
        ax.plot([x0, x1], [y0, y1], c=col_edges, zorder=5, alpha=alpha_edges, linewidth=linewidth)
    if disp_id:
        offset=0.02
        for i, (x,y) in enumerate(coords):
            plt.text(x-offset, y-offset, str(i), zorder=15)
    if aspect is not None:
        ax.set_aspect(aspect)
    if ax_none:
        return fig, ax

def plot_network_distances(coords, pairs, distances, labels=None,
                           color_mapper=None, legend=True,
                           col_nodes=None, cmap_nodes=None, marker=None, size_nodes=None, 
                           cmap_edges='viridis', alpha_edges=0.7, linewidth=None,
                           figsize=(15, 15), ax=None, aspect='equal', **kwargs):
    """
    Plot a network with edges colored by their length.

    Parameters
    ----------
    coords : ndarray
        Coordinates of points where each column corresponds to an axis (x, y, ...)
    pairs : ndarray
        The (n_pairs x 2) array of neighbors indices.
    distances : array
        Distances between each pair of neighbors.
    labels: panda series
        The nodes' labels from which they are colored.
    color_mapper: dict
        Maps each label to its color. Computed if not provided.
    col_nodes : str of matplotlib compatible color, optional
        Color of nodes. The default is None.
    cmap_nodes: list
        List of hexadecimal colors for nodes attributes.
    marker : str, optional
        Marker used to display nodes. The default is None.
    size_nodes : float, optional
        Size of nodes. The default is None.
    cmap_edges : str of matplotlib.colormap, optional
        Colormap of edges. The default is 'viridis'.
    alpha_edges : float, optional
        Tansparency of edges. The default is 0.7.
    linewidth : float, optional
        Width of edges. The default is None.
    figsize : (float, float), default: :rc:`figure.figsize`
        Width, height in inches. The default is (15, 15).
    ax : matplotlib ax object, optional
        If provided, the plot is displayed in ax. The default is None.
    aspect : str, optional
        Proportions of the figure. The default is None.
    **kwargs : TYPE
        labels of nodes.

    Returns
    -------
    None or (fig, ax) if not provided in parameters.

    """
    
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
        # color nodes with manual colors
        if color_mapper is None:
            if cmap_nodes is None:
                att_colors = sns.color_palette('muted').as_hex() 
            color_mapper = dict(zip(uniq, att_colors))
        for label in uniq:
            select = labels == label
            color = color_mapper[label]
            ax.scatter(coords[select,0], coords[select,1], c=color, label=label,
                       marker=marker, s=size_nodes, zorder=10, **kwargs)
        if legend:
            plt.legend()
    else:
        ax.scatter(coords[:,0], coords[:,1], c=col_nodes, cmap=cmap_nodes, 
                   marker=marker, s=size_nodes, zorder=10, **kwargs)
    # plot edges
    scaled_dist, min_dist, max_dist = rescale(distances, return_extrema=True)
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
    for pair, dist in zip(pairs[:,:], scaled_dist):
        [x0, y0], [x1, y1] = coords[pair]
        ax.plot([x0, x1], [y0, y1], c=cmap(dist), zorder=0, alpha=alpha_edges, linewidth=linewidth)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='vertical', label='Distance')
    # TODO: plot many lines more efficiently check
    # from https://stackoverflow.com/a/50029441
    # https://matplotlib.org/gallery/shapes_and_collections/line_collection.html#sphx-glr-gallery-shapes-and-collections-line-collection-py
    
    if aspect is not None:
        ax.set_aspect(aspect)
    if ax_none:
        return fig, ax

def showim(image, figsize=(9,9), ax=None, **kwargs):
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

def categorical_to_integer(l):
    uniq = set(l)
    nb_uniq = len(uniq)
    mapping = dict(zip(uniq, range(nb_uniq)))
    converted = [mapping[x] for x in l]
    return converted

def flatten_categories(nodes, att):
    # the reverse operation is 
    # nodes = nodes.join(pd.get_dummies(nodes['nodes_class']))
    return nodes.loc[:, att].idxmax(axis=1)

def coords_to_df(coords, columns=None):
    """
    Convert an array of coordinates of nodes into a dataframe.

    Parameters
    ----------
    coords : ndarray
        Coordinates of points with columns corresponding to axes ('x', 'y', ...)
    columns : Index or array-like
        Column labels to use for resulting frame. Will default to
        ['x0', 'x1',..., 'xn'] if no column labels are provided.

    Returns
    -------
    nodes : dataframe
        Coordinates of nodes indicated by 'x', 'y' or other if required.
    """
    nb_dim = coords.shape[1]
    if columns is None:
        if nb_dim == 2:
            columns = ['x', 'y']
        elif nb_dim == 3:
            columns = ['x', 'y', 'y']
        else:
            columns = ['x'+str(i) for i in range(nb_dim)]
    
    nodes = pd.DataFrame(data=coords, columns=columns)
    return nodes

def pairs_to_df(pairs, columns=['source', 'target']):
    """
    Convert an array of pairs of nodes into a dataframe

    Parameters
    ----------
    pairs : ndarray
        The (n_pairs x 2) array of neighbors indices.
    columns : Index or array-like
        Column labels to use for resulting frame. Default is ['source', 'target']

    Returns
    -------
    edges : dataframe
        Edges indicated by the nodes 'source' and 'target' they link.
    """

    edges = pd.DataFrame(data=pairs, columns=columns)
    return edges
       
def to_NetworkX(nodes, edges, attributes=None):
    """
    Convert tysserand network representation to a NetworkX network object

    Parameters
    ----------
    nodes : ndarray or dataframe
        Coordinates of points with columns corresponding to axes ('x', 'y', ...)
    edges : ndarray or dataframe
        The pairs of nodes given by their indices.
    attributes : dataframe
        Attributes of nodes to be added in NetworkX. Default is None.

    Returns
    -------
    G : NetworkX object
        The converted network.
    """
    
    import networkx as nx
    # convert to dataframe if numpy array
    if isinstance(nodes, np.ndarray):
        nodes = coords_to_df(nodes)
    if isinstance(edges, np.ndarray):
        edges = pairs_to_df(edges)
    
    G = nx.from_pandas_edgelist(edges)
    if attributes is not None:
        for col in attributes.columns:
            # only for glm extension file:
            # nx.set_node_attributes(G, attributes[col].to_dict(), col.replace('+','AND')) 
            nx.set_node_attributes(G, attributes[col].to_dict(), col)
    return G

def to_iGraph(nodes, edges, attributes=None):
    """
    Convert tysserand network representation to an iGraph network object

    Parameters
    ----------
    nodes : ndarray or dataframe
        Coordinates of points with columns corresponding to axes ('x', 'y', ...)
    edges : ndarray or dataframe
        The pairs of nodes given by their indices.
    attributes : dataframe
        Attributes of nodes to be added in NetworkX. Default is None.

    Returns
    -------
    G : iGraph object
        The converted network.
    """
    
    import igraph as ig
    # convert to dataframe if numpy array
    if isinstance(nodes, np.ndarray):
        nodes = coords_to_df(nodes)
    if isinstance(edges, np.ndarray):
        edges = pairs_to_df(edges)
    
    # initialize empty graph
    G = ig.Graph()
    # add all the vertices
    G.add_vertices(nodes.shape[0])
    # add all the edges
    G.add_edges(edges.values)
    # add attributes
    if attributes is not None:
        for col in attributes.columns:
            att = attributes[col].values
            if isinstance(att[0], str):
                att = categorical_to_integer(att)
            G.vs[col] = att
    return G

def add_to_AnnData(coords, pairs, adata):
    """    
    Convert tysserand network representation to sparse matrices
    and add them to an AnnData (Scanpy) object.

    Parameters
    ----------
    nodes : ndarray
        Coordinates of points with columns corresponding to axes ('x', 'y', ...)
    edges : ndarray
        The pairs of nodes given by their indices.
    adata : AnnData object
        An object dedicated to single-cell data analysis.
    """
    
    # convert arrays to sparse matrices
    n_cells = adata.shape[0]
    connect = np.ones(pairs.shape[0], dtype=np.int8)
    sparse_connect = csr_matrix((connect, (pairs[:,0], pairs[:,1])), shape=(n_cells, n_cells), dtype=np.int8)
    distances = distance_neighbors(coords, pairs)
    sparse_dist = csr_matrix((distances, (pairs[:,0], pairs[:,1])), shape=(n_cells, n_cells), dtype=np.float)
    
    # add to AnnData object
    adata.obsp['connectivities'] = sparse_connect
    adata.obsp['distances'] = sparse_dist
    adata.uns['neighbors'] = {'connectivities_key': 'connectivities', 
                              'distances_key': 'distances', 
                              'params': {'method': 'delaunay', 
                                         'metric': 'euclidean', 
                                         'edge_trimming': 'percentile 99'}}
