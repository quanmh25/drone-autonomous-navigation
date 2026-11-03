"""Pure distance measurements from LiDAR scans."""

import math

import numpy as np


def sector(msg, start, end):
    """Return nearest metres; positive infinity is clear, invalid rays are unknown."""
    if (msg is None or not msg.ranges or not math.isfinite(msg.angle_increment) or
            msg.angle_increment == 0):
        return None
    angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
    ranges = np.asarray(msg.ranges)
    selected = ranges[(angles >= math.radians(start)) & (angles <= math.radians(end))]
    if not selected.size:
        return None
    valid = selected[np.isfinite(selected) & (selected > 0) & (selected <= msg.range_max)]
    # Keep even below-minimum positive returns as close obstacles.
    if valid.size:
        return float(valid.min())
    return math.inf if np.isposinf(selected).all() else None
