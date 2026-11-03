"""Locate map storage and export odom-aligned occupancy grids."""

import math
import os
from pathlib import Path as FilePath
import tempfile

from ament_index_python.packages import get_package_share_directory
import numpy as np
import yaml


def map_directory():
    """Prefer source-package map storage, including with copied colcon installs."""
    share = FilePath(get_package_share_directory('drone_lidar_control')).resolve()
    for root in (*FilePath(__file__).resolve().parents, *share.parents,
                 FilePath.cwd(), *FilePath.cwd().parents):
        for package in (root, root / 'src' / 'drone_lidar_control'):
            if (package / 'package.xml').is_file() and (package / 'map').is_dir():
                return str(package / 'map')
    return str(share / 'map')


def save_map(msg, filename, transform):
    """Freeze map geometry in odom and commit a lossless snapshot last."""
    h, w, res = msg.info.height, msg.info.width, msg.info.resolution
    if min(h, w, res) <= 0 or len(msg.data) != h*w:
        raise ValueError('Invalid occupancy grid')
    data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
    if not np.any(data >= 0):
        raise ValueError('Map has no observed cells')
    p, q = msg.info.origin.position, msg.info.origin.orientation
    t, r = transform.translation, transform.rotation
    if abs(r.x) > 1e-4 or abs(r.y) > 1e-4:
        raise ValueError('Expected planar map -> odom transform')
    yaw = math.atan2(2*(r.w*r.z+r.x*r.y), 1-2*(r.y*r.y+r.z*r.z))
    grid_yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
    origin = [t.x+math.cos(yaw)*p.x-math.sin(yaw)*p.y,
              t.y+math.sin(yaw)*p.x+math.cos(yaw)*p.y, yaw+grid_yaw]
    path = FilePath(filename).expanduser().with_suffix('.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(data.shape, 205, dtype=np.uint8)
    image[(data >= 0) & (data < 50)] = 254
    image[data >= 50] = 0
    path.with_suffix('.pgm').write_bytes(f'P5\n{w} {h}\n255\n'.encode()+np.flipud(image).tobytes())
    path.with_suffix('.yaml').write_text(yaml.safe_dump({
        'image': path.with_suffix('.pgm').name, 'resolution': float(res),
        'origin': origin, 'negate': 0, 'occupied_thresh': .65, 'free_thresh': .196}))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, data=data, resolution=res, origin=origin, frame='odom')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)
