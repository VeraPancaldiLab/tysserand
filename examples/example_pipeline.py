# -*- coding: utf-8 -*-

#%% Voronoi tesselation

# With a very simple network
coords = make_simple_coords()
pairs = build_voronoi(coords, trim_dist=False)
distances = distance_neighbors(coords, pairs)
plot_network_distances(coords, pairs, distances)
dist_threshold = 120
select = distances < dist_threshold
pairs = pairs[select,:]
plot_network(coords, pairs)


# With a very simple network
img = plt.imread("../data/mIF_WSI_tile/tile.png")
nodes = pd.read_csv(r"../data/mIF_WSI_tile/nodes.csv", usecols=['x','y', 'marker'])

coords = nodes.loc[:,['x','y']].values
pairs = build_voronoi(coords, trim_dist=False)
distances = distance_neighbors(coords, pairs)
plot_network_distances(coords, pairs, distances)
dist_threshold = 120
select = distances < dist_threshold
pairs = pairs[select,:]

# make colors for nodes
# marker --> DAPI 5060C, FITC+TxRed-A-2, FITC+TxRed-A-1 
# class_colors --> blue, orange, green
class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
classes = list(nodes['marker'].unique())
dico_col = {classes[0]:class_colors[0],
            classes[1]:class_colors[1],
            classes[2]:class_colors[2]}
colors = []
for cl in nodes['marker']:
    colors.append(dico_col[cl])
# superimpose network to mIF image
fig, ax = showim(img)
plot_network(coords, pairs, col_nodes=colors, col_edges='w', ax=ax)


#%% Neirest Neighbors

# With a very simple network
coords = make_simple_coords()
pairs = build_NN(coords)
distances = distance_neighbors(coords, pairs)
plot_network_distances(coords, pairs, distances)
dist_threshold = 120
select = distances < dist_threshold
pairs = pairs[select,:]
plot_network(coords, pairs)
# We can notice that the Neirest Neighbors approach outputs edges that we would 
# like to discard

# With a very simple network
img = plt.imread("../data/mIF_WSI_tile/tile.png")
nodes = pd.read_csv(r"../data/mIF_WSI_tile/nodes.csv", usecols=['x','y', 'marker'])

coords = nodes.loc[:,['x','y']].values
pairs = build_NN(coords)
distances = distance_neighbors(coords, pairs)
plot_network_distances(coords, pairs, distances)
dist_threshold = 120
select = distances < dist_threshold
pairs = pairs[select,:]

# make colors for nodes
# marker --> DAPI 5060C, FITC+TxRed-A-2, FITC+TxRed-A-1 
# class_colors --> blue, orange, green
class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
classes = list(nodes['marker'].unique())
dico_col = {classes[0]:class_colors[0],
            classes[1]:class_colors[1],
            classes[2]:class_colors[2]}
colors = []
for cl in nodes['marker']:
    colors.append(dico_col[cl])
# superimpose network to mIF image
fig, ax = showim(img)
plot_network(coords, pairs, col_nodes=colors, col_edges='w', ax=ax)


#%% Within Radius

# With a very simple network
coords = make_simple_coords()
pairs = build_within_radius(coords, r=60)
distances = distance_neighbors(coords, pairs)
plot_network_distances(coords, pairs, distances)

# The Within Radius approach outputs either a completely connected network
# with a higly connected cluster, or no highly connected cluster but 
# an isolated point

# With a very simple network
img = plt.imread("../data/mIF_WSI_tile/tile.png")
nodes = pd.read_csv(r"../data/mIF_WSI_tile/nodes.csv", usecols=['x','y', 'marker'])

coords = nodes.loc[:,['x','y']].values
pairs = build_within_radius(coords, r=60)
distances = distance_neighbors(coords, pairs)
plot_network_distances(coords, pairs, distances)
dist_threshold = 120
select = distances < dist_threshold
pairs = pairs[select,:]

# make colors for nodes
# marker --> DAPI 5060C, FITC+TxRed-A-2, FITC+TxRed-A-1 
# class_colors --> blue, orange, green
class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
classes = list(nodes['marker'].unique())
dico_col = {classes[0]:class_colors[0],
            classes[1]:class_colors[1],
            classes[2]:class_colors[2]}
colors = []
for cl in nodes['marker']:
    colors.append(dico_col[cl])
# superimpose network to mIF image
fig, ax = showim(img)
plot_network(coords, pairs, col_nodes=colors, col_edges='w', ax=ax)

# The Within Radius approach is not adapted when nodes density varies, there
# are some areas with few edges and other areaas with (too) many edges


#%% Contacting regions

# With simple artifical data
W = 100
H = 100

coords = np.array([[15, 25],
                   [35, 25],
                   [55, 25],
                   [35, 45]])

binary_im = np.full((H, W), False)
r = 10
for a, b in coords:
    y, x = np.ogrid[-a:H-a, -b:W-b]
    circle = x*x + y*y <= r*r
    binary_im[circle] = True

distance = ndi.distance_transform_edt(binary_im)
local_maxi = feature.peak_local_max(distance, indices=False,
                                    min_distance=5)
markers = measure.label(local_maxi)
masks = segmentation.watershed(-distance, markers, mask=binary_im)
showim(masks)

pairs = build_contacting(masks)-1
distances = distance_neighbors(coords, pairs)
plot_network_distances(coords, pairs, distances)


# With nuclei image
img = plt.imread("../data/mIF_WSI_tile/nuclei_grey.png")

thresholds = filters.threshold_multiotsu(img, classes=3)
# regions = np.digitize(img, bins=thresholds)
binary_im = img > thresholds[0]
showim(binarized)
distance = ndi.distance_transform_edt(binary_im)
local_maxi = feature.peak_local_max(distance, indices=False,
                                    min_distance=5)
markers = measure.label(local_maxi)
masks = segmentation.watershed(-distance, markers, mask=binary_im)
showim(color.label2rgb(masks, bg_label=0))

pairs = build_contacting(masks)-1
distances = distance_neighbors(coords, pairs)
plot_network_distances(coords, pairs, distances)