"""Export RGB-D occupancy maps in a stable odometry frame."""

import math
import os
from pathlib import Path as FilePath
import tempfile

from nav_msgs.msg import OccupancyGrid
import numpy as np
import yaml

from .geometry import yaw_of


def default_map_directory(share_directory):
    """Prefer workspace source config with both copied and symlink installs."""
    share = FilePath(share_directory).resolve()
    for root in (FilePath.cwd(), *FilePath.cwd().parents, *share.parents):
        package = root / 'src' / 'drone_vision_control'
        if (package / 'package.xml').is_file() and (package / 'config').is_dir():
            return str(package / 'config')
    return str(share / 'config')


def save_snapshot(msg, filename, transform=(0., 0., 0.), frame='odom'):
    """Freeze the grid in a stable frame; commit the lossless NPZ file last."""
    destination = FilePath(filename).expanduser().with_suffix('.npz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    h, w = msg.info.height, msg.info.width
    if h <= 0 or w <= 0 or msg.info.resolution <= 0 or len(msg.data) != h * w:
        raise ValueError('Invalid occupancy grid')
    data = np.asarray(msg.data, dtype=np.int8).reshape(h, w)
    if not np.any(data >= 0):
        raise ValueError('Map contains no observed cells')
    tx, ty, yaw = transform
    p, q = msg.info.origin.position, msg.info.origin.orientation
    grid_yaw = yaw_of(q)
    origin = [tx + math.cos(yaw) * p.x - math.sin(yaw) * p.y,
              ty + math.sin(yaw) * p.x + math.cos(yaw) * p.y, yaw + grid_yaw]
    image = np.full((h, w), 205, dtype=np.uint8)
    image[(data >= 0) & (data < 50)] = 254
    image[data >= 50] = 0
    with destination.with_suffix('.pgm').open('wb') as stream:
        stream.write(f'P5\n{w} {h}\n255\n'.encode())
        stream.write(np.flipud(image).tobytes())
    metadata = {'image': destination.with_suffix('.pgm').name,
                'resolution': float(msg.info.resolution), 'origin': origin,
                'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196,
                'mode': 'trinary', 'frame_id': frame}
    destination.with_suffix('.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.npz',
                                         delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, data=data, resolution=float(msg.info.resolution),
                                origin=np.asarray(origin), frame_id=frame)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination)


def project_tilted_grid(msg, transform):
    """Project the full 3D grid transform onto odom XY, retaining obstacles."""
    q = transform.rotation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(quat).all() or norm < 1e-9:
        raise ValueError('Invalid map transform quaternion')
    x, y, z, w = quat / norm

    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    yaw = yaw_of(msg.info.origin.orientation)
    grid_rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                              [math.sin(yaw), math.cos(yaw)]])
    affine = rotation[:2, :2] @ grid_rotation

    if abs(np.linalg.det(affine)) < .1:
        raise ValueError('Map plane is almost vertical; cannot form a useful ground map')
    p, t = msg.info.origin.position, transform.translation
    offset = (rotation @ np.array([p.x, p.y, p.z]))[:2] + [t.x, t.y]
    res, h, width = msg.info.resolution, msg.info.height, msg.info.width
    data = np.asarray(msg.data, dtype=np.int16).reshape(h, width)
    corners = np.array([[0, 0], [width, 0], [0, h], [width, h]]) * res
    projected = corners @ affine.T + offset
    low = np.floor(projected.min(axis=0) / res) * res
    shape = np.maximum(1, np.ceil((projected.max(axis=0) - low) / res).astype(int))
   
    if int(shape[0]) * int(shape[1]) > 10_000_000:
        raise ValueError('Projected grid is too large')
    output = np.full((shape[1], shape[0]), -1, dtype=np.int16)
    inverse = np.linalg.inv(affine)
    
    for row in range(output.shape[0]):
        centers = np.column_stack((low[0] + (np.arange(shape[0]) + .5) * res,
                                   np.full(shape[0], low[1] + (row + .5) * res)))
        source = np.floor(((centers - offset) @ inverse.T) / res).astype(int)
        valid = ((source[:, 0] >= 0) & (source[:, 0] < width) &
                 (source[:, 1] >= 0) & (source[:, 1] < h))
        output[row, valid] = data[source[valid, 1], source[valid, 0]]
    # A tilted occupied cell can be narrower than one output pixel. Rasterize
    # its whole projected bounding box conservatively so it cannot disappear.
    
    for sy, sx in np.argwhere(data >= 50):
        vertices = (np.array([[sx, sy], [sx+1, sy], [sx, sy+1], [sx+1, sy+1]]) * res)
        pixels = ((vertices @ affine.T + offset) - low) / res
        first = np.maximum(0, np.floor(pixels.min(axis=0) + 1e-9).astype(int))
        last = np.minimum(shape, np.ceil(pixels.max(axis=0) - 1e-9).astype(int))
        output[first[1]:last[1], first[0]:last[0]] = 100
   
    result = OccupancyGrid()
    result.header.frame_id = 'odom'
    result.info.resolution = float(res)
    result.info.width, result.info.height = int(shape[0]), int(shape[1])
    result.info.origin.position.x, result.info.origin.position.y = map(float, low)
    result.info.origin.orientation.w = 1.
    result.data = output.ravel().tolist()
    return result


def save_odom_snapshot(grid, filename, transform=None):
    """Apply a map-to-odom transform and export the frozen occupancy grid."""
    planar = (0., 0., 0.)
    if grid.header.frame_id != 'odom':
        if transform is None:
            raise ValueError('A map-to-odom transform is required')
        q, p = transform.rotation, transform.translation
        if abs(q.x) > 1e-6 or abs(q.y) > 1e-6:
            grid = project_tilted_grid(grid, transform)
        else:
            planar = (p.x, p.y, yaw_of(q))
    return save_snapshot(grid, filename, planar)
