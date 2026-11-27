# API reference

Every public function is available from the package namespace
(`import tysserand as ty`, then `ty.build_delaunay(...)`); the pages below
group them by module.

| Module | Content |
| ------ | ------- |
| [build](build.md) | coordinates to networks: Delaunay, k-nearest neighbors, radial distance, lattice, solitary nodes |
| [contacting](contacting.md) | segmented images to networks of contacting areas, tiled parallel version |
| [convert](convert.md) | CSR matrices, edge pairs and DataFrames, attribute matrices |
| [distances](distances.md) | edge distances and trimming thresholds |
| [io](io.md) | networks saved as `.npz` files, migration of edge-list files |
| [plot](plot.md) | networks and edge distances plots |
| [interop](interop.md) | NetworkX, iGraph and AnnData bridges |
| [annotation](annotation.md) | interactive visualization and annotation with napari |
| [scoring](scoring.md) | comparison of reconstructed networks to a ground truth |
| [generators](generators.md) | synthetic nodes and segmented images |
