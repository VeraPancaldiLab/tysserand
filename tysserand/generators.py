"""
Synthetic nodes and segmented images, for tests and tutorials.
"""
import numpy as np

__all__ = [
    'make_simple_coords',
    'make_random_nodes',
    'make_random_tiles',
]


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


def make_random_tiles(sx=500, sy=500, sz=0, nb=50, noise_sigma=None,
                      regular=True, double_pattern_y=False, double_pattern_z=False, 
                      assym_y=True, assym_z=True, return_image=False):
    """
    Build contacting areas similar to cell segmentation in tissues.

    Parameters
    ----------
    sx : int, optional
        Size of the image on the x axis. The default is 500.
    sy : int, optional
        Size of the image on the y axis. The default is 500.
    sz : int, optional
        Size of the image on the z axis. The default is 0, which
        implies generating a 2D image.
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
    >>> label_cmap = mpl.colormaps.get_cmap('Set2')(range(8))
    >>> showim(color.label2rgb(masks, bg_label=0, colors=label_cmap), origin='lower')

    """
    from skimage import segmentation
    
    if sz == 0:
        image = np.zeros((sy, sx))
        # to overcome an issue with odd nb:
        nb = int(np.ceil(nb / 2) * 2)
        
        if regular:
            x = np.linspace(start=0, stop=sx-1, num=nb, dtype=int)
            x = np.hstack((x[::2], x[1::2]))
            if assym_y:
                nb = nb*2
            y = np.linspace(start=0, stop=sy-1, num=nb, dtype=int)
            if double_pattern_y:
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
    else:
        # make 3D simulation
        image = np.zeros((sz, sy, sx))
        # to overcome an issue with odd nb:
        nb = int(np.ceil(nb / 2) * 2)
        
        if regular:
            x = np.linspace(start=0, stop=sx-1, num=nb, dtype=int)
            x = np.hstack((x[::2], x[1::2]))
            if assym_y:
                nb_y = nb*2
            y = np.linspace(start=0, stop=sy-1, num=nb_y, dtype=int)
            if assym_z:
                nb_z = nb*2
            z = np.linspace(start=0, stop=sz-1, num=nb_z, dtype=int)
            if double_pattern_y:
                y = np.hstack((y[::2], y[1::2]))
            if double_pattern_z:
                z = np.hstack((z[::2], z[1::2]))
            x_id = np.tile(x, y.size//2)
            y_id = np.repeat(y, x.size//2)
            z_id = np.repeat(z, x.size//2)
        else:
            x_id = np.random.randint(sx, size=nb)
            y_id = np.random.randint(sy, size=nb)
            z_id = np.random.randint(sz, size=nb)
            
        if noise_sigma is None:
            print("For 3D simulations noise_sigma needs to be > 0")
            print("Setting noise_sigma to 1")
            noise_sigma = 1
        # x
        x_id = x_id + np.random.normal(loc=0.0, scale=noise_sigma, size=x_id.size)
        x_id[x_id<0] = 0
        x_id[x_id>sx-1] = sx-1
        x_id = np.round(x_id).astype(int)
        # y
        y_id = y_id + np.random.normal(loc=0.0, scale=noise_sigma, size=y_id.size)
        y_id[y_id<0] = 0
        y_id[y_id>sy-1] = sy-1
        y_id = np.round(y_id).astype(int)
        # z
        z_id = z_id + np.random.normal(loc=0.0, scale=noise_sigma, size=z_id.size)
        z_id[z_id<0] = 0
        z_id[z_id>sz-1] = sz-1
        z_id = np.round(z_id).astype(int)
            
        coords = np.vstack((x_id, y_id, z_id)).T
        image[z_id, y_id, x_id] = 1
        masks = segmentation.watershed(-image)

    if return_image:
        return coords, masks, image
    else:
        return coords, masks
