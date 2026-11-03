"""Planar orientation helpers for RGB-D navigation."""

import math

import numpy as np


def yaw_of(q):
    """Extract planar yaw from a ROS quaternion."""
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(angle):
    """Wrap an angular error."""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(x, limit):
    return float(np.clip(x, -limit, limit))


def goal_geometry(position_xy, waypoint):
    """Return distance and heading to the current waypoint."""
    goal = np.asarray(waypoint, dtype=float)
    delta = goal - np.asarray(position_xy, dtype=float)
    return float(np.linalg.norm(delta)), math.atan2(delta[1], delta[0])


def on_goal_line(position_xy, leg_start, waypoint, hit_goal_distance):
    """Check return to the goal segment with enough progress past the obstacle."""
    a = np.asarray(leg_start, dtype=float)
    b = np.asarray(waypoint, dtype=float)
    p = np.asarray(position_xy, dtype=float)
    v = b - a
    length = np.linalg.norm(v)

    if length < 1e-6:
        return False

    along = np.dot(p - a, v) / (length * length)
    if not 0.0 <= along <= 1.0:
        return False

    cross = abs(v[0]*(p[1]-a[1]) - v[1]*(p[0]-a[0])) / length
    return cross < 0.30 and goal_geometry(position_xy, waypoint)[0] < hit_goal_distance - 0.35

