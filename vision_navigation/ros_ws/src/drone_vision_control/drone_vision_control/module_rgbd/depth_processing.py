"""Depth image measurements for reactive RGB-D navigation."""

import math

import numpy as np


def depth_in_metres(msg, bridge):
    """Convert a supported ROS depth image to a float32 array in metres."""
    if msg.encoding not in ('16UC1', '32FC1'):
        raise ValueError(f'Unsupported depth encoding: {msg.encoding}')
    depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
    if msg.encoding == '16UC1':
        depth /= 1000.0
    return depth


def depth_distance(depth, front_limit):
    """Estimate conservative forward clearance from axial depth in metres."""
    if depth is None or depth.ndim != 2 or depth.size == 0:
        return None

    h, w = depth.shape
    roi = depth[int(h * 0.20):int(h * 0.55), int(w * 0.30):int(w * 0.70)]
    valid = roi[np.isfinite(roi) & (roi > 0.0) & (roi <= 10.0)]

    # Preserve very close returns; they must never become "unknown".
    close = valid[valid < front_limit]
    if close.size >= max(3, roi.size * 0.005):
        return float(np.percentile(close, 10))

    if valid.size >= max(1, roi.size * 0.05):
        return float(np.percentile(valid, 10))

    if np.count_nonzero(np.isposinf(roi)) > roi.size * 0.5:
        return 10.0

    return None


def right_wall_geometry(depth):
    """Estimate wall distance and tangent from the side camera's axial depth."""
    if depth is None or depth.ndim != 2:
        return None
    h, w = depth.shape
    if h < 4 or w < 16:
        return None
    # Matches right_rgbd horizontal_fov in drone_camera.xacro. For a right
    # facing camera, optical X points toward body -X, optical Z toward -Y.
    fx = w / (2.0 * math.tan(1.0472 / 2.0))
    xs, ys = [], []
    for col in range(int(w * 0.15), int(w * 0.85)):
        samples = depth[int(h * 0.3):int(h * 0.55), col]
        valid = samples[np.isfinite(samples) & (samples > 0.15) & (samples < 3.)]
        if valid.size < max(2, samples.size // 2):
            continue
        z = float(np.median(valid))
        xs.append(-(col - (w - 1) / 2.0) * z / fx)
        ys.append(-z)
    if len(xs) < 12 or np.ptp(xs) < 0.15:
        return None
    x, y = np.asarray(xs), np.asarray(ys)
    design = np.column_stack((x, np.ones_like(x)))
    slope, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    inliers = np.abs(y - (slope * x + offset)) < 0.08
    if np.count_nonzero(inliers) < max(12, len(xs) * 0.75):
        return None  # Multiple surfaces or a corner: do not fit across them.
    slope, offset = np.linalg.lstsq(design[inliers], y[inliers], rcond=None)[0]
    if offset >= 0. or abs(slope) > 1.5:
        return None
    return abs(offset) / math.hypot(1., slope), math.atan(slope)
