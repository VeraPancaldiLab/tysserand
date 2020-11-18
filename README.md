# tysserand

A library for fast spatial network reconstruction.  

*tysserand* is a Python library to reconstruct spatial networks from spatially resolved omics experiments. It is intended as a common tool where the bioinformatics community can add new methods to reconstruct networks, choose appropriate parameters, clean resulting networks and pipe data to other libraries.  
You can find the preprint and supplementary information on [BioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.16.385377v1).  
A turorial is coming very soon! (today actually :-) )

## Implemented methods

![Set of nodes](./images/publication_figures/mIF - nodes positions.png)

### Voronoi tessellation

This methods builds virtual cells centered arround each node and contacting each other to fully tile the space occupyied by the nodes. Edges are drawn between the nodes of contacting tiles.

![Edge lengths with *Delaunay* reconstruction](./images/publication_figures/mIF - Delaunay distances.png)
![Trimmed network](./images/publication_figures/mIF - Delaunay network.png)
![Network overlay on original tissue image](./images/publication_figures/mIF - Delaunay superimposed.png)

### K Nearest Neighbors

Each node is linked with its k nearest neighbors. It is the most common method used in single cell publications, althought it produces artifact well visible on simple 2D networks.

![Edge lengths with *k-nearest neighbors* reconstruction](./images/publication_figures/mIF - knn distances.png)

### Within radius

Each node is linked to nodes closer than a threshold distance D, that is to say each node is linked to all nodes in a circle of radius D.

![Edge lengths with *radial distance neighbors* reconstruction](./images/publication_figures/mIF - rdn distances.png)

### Contacting areas

Nodes are the center of detected objects (like after cell segmentation) and they are linked if their respective areas are in contact or closer than a given distance threshold.

![Examplary data for the *Contacting areas* reconstruction](./images/publication_figures/generated-tissue-interger-masks.png)
![Edge lengths with *Contacting areas* reconstruction](./images/publication_figures/generated-tissue-cell-contact-superimposition.png)
