# tysserand

A library for spatial network reconstruction


## Implementeed methods

![Set of nodes](./images/network_nodes.png)

### Voronoi tessellation

This methods builds virtual cells centered arround each node and contacting each other to fully tile the space occupyied by the nodes. Edges are drawn between the nodes of contacting tiles.

![Edge lengths with *Voronoi* reconstruction](./images/network_Voronoi_distances.png)
![Trimmed network](./images/network_trimmed.png)
![Network overlay on original tissue image](./images/network_image_overlay.png)

### K Nearest Neighbors

Each node is linked with its k nearest neighbors. It is the most common method used in single cell publications, althought it produces artifact well visible on simple 2D networks.

![Edge lengths with *kNN* reconstruction](./images/network_kNN_distances.png)

### Within radius

Each node is linked to nodes closer than a threshold distance D, that is to say each node is linked to all nodes in a circle of radius D.

![Edge lengths with *Within radius* reconstruction](./images/network_within_radius_distances.png)

### Contacting areas

Nodes are the center of detected objects (like after cell segmentation) and they are linked if their respective areas are in contact or closer than a given distance threshold.

![Examplary data for the *Contacting areas* reconstruction](./images/contacting_areas.png)
![Edge lengths with *Contacting areas* reconstruction](./images/contacting_areas_network_higher_distances.png)

