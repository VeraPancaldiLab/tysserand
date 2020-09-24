# -*- coding: utf-8 -*-

coords = make_simple_coords()
all_pairs = build_voronoi(coords, trim_dist=False)
distances = distance_neighbors(coords, all_pairs)
plot_network_distances(coords, all_pairs, distances)

plot_network(coords, all_pairs)
