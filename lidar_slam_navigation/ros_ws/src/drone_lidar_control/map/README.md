# Saved LiDAR maps

Mapping saves `arena_latest.npz` here after arrival, with `.yaml` and `.pgm`
exports. Wait for `Map saved:` before stopping mapping and launching
`astar.launch.py`. Both launches use this directory by default.

The next successful mapping run replaces the previous snapshot. Use
`save_map_path` and `map_file` launch arguments to select another file.
