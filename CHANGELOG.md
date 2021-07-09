# Changelog

## v0.2.0

v0.2.0 implements 2 new methods, corrects a bug and improves the project organization.

**New features**

- The `build_contacting_nn` method allows to link segmented objects that are close to each other, then to link the remaining "lonely" objects to their nearest neighbors in order to avoid too many unconnect nodes in the resulting network
- The `build_contacting` method has a parallel version implemented with Dask, in which big images are splitted into an optimal numberof tiles depending on the number of cores.

**Fixed bugs:**

- The `build_knn` method was actually building networks with (k-1) neighbors as the first k was 'oneself' and was discarded. Now it build networks with k neighboring nodes.

**Project organization**

- Added poetry project configuration files `pyproject.toml` and `poetry.lock` for better reproducibility, see [here](https://modelpredict.com/python-dependency-management-tools) for some discussion. A guide for easy installation with all commands coming soon :)
- This very CHANGELOG file!
