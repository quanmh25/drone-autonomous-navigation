"""Fly shortest safe grid paths on saved odom-aligned SLAM snapshots."""

import heapq
import math
from pathlib import Path as FilePath
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def wrap(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def map_directory():
    """Prefer source-package map storage, including with copied colcon installs."""
    share = FilePath(get_package_share_directory('drone_lidar_control')).resolve()
    for root in (*FilePath(__file__).resolve().parents, *share.parents,
                 FilePath.cwd(), *FilePath.cwd().parents):
        for package in (root, root / 'src' / 'drone_lidar_control'):
            if (package / 'package.xml').is_file() and (package / 'map').is_dir():
                return str(package / 'map')
    return str(share / 'map')


def load_map(filename):
    """Restore exact occupancy values and the frozen odom origin."""
    with np.load(FilePath(filename).expanduser(), allow_pickle=False) as archive:
        data, origin = archive['data'], archive['origin']
        res, frame = float(archive['resolution']), str(archive['frame'])
    if (data.ndim != 2 or not data.size or not np.isfinite(res) or res <= 0 or
            origin.shape != (3,) or not np.isfinite(origin).all() or frame != 'odom' or
            not np.isfinite(data).all() or np.any((data < -1) | (data > 100))):
        raise ValueError('Invalid odom map snapshot')
    msg = OccupancyGrid()
    msg.header.frame_id = frame
    msg.info.height, msg.info.width = data.shape
    msg.info.resolution = res
    msg.info.origin.position.x, msg.info.origin.position.y = map(float, origin[:2])
    msg.info.origin.orientation.z = math.sin(float(origin[2])/2)
    msg.info.origin.orientation.w = math.cos(float(origin[2])/2)
    msg.data = data.astype(np.int8).ravel().tolist()
    return msg


def grid_cell(point, origin, resolution):
    """Convert odom XY to a cell, including a rotated map origin."""
    dx, dy = np.asarray(point)-origin[:2]
    c, s = math.cos(origin[2]), math.sin(origin[2])
    return math.floor((c*dx+s*dy)/resolution), math.floor((-s*dx+c*dy)/resolution)


def blocked_grid(data, resolution, clearance):
    """Inflate obstacles, unknown cells and map boundaries by vehicle clearance."""
    occupied = (data < 0) | (data >= 50)
    blocked = occupied.copy()
    radius = math.ceil(clearance/resolution)
    h, w = data.shape
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if math.hypot(dx, dy)*resolution > clearance+resolution*math.sqrt(2):
                continue
            sy, ey = max(0, -dy), min(h, h-dy)
            sx, ex = max(0, -dx), min(w, w-dx)
            if ey > sy and ex > sx:
                blocked[sy+dy:ey+dy, sx+dx:ex+dx] |= occupied[sy:ey, sx:ex]
    if radius:
        blocked[:radius] = blocked[-radius:] = True
        blocked[:, :radius] = blocked[:, -radius:] = True
    return blocked


def astar(blocked, start, goal):
    h, w = blocked.shape
    directions = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]

    def free(x, y):
        return 0 <= x < w and 0 <= y < h and not blocked[y, x]

    def heuristic(x, y):
        dx, dy = abs(x-goal[0]), abs(y-goal[1])
        return max(dx, dy) + (math.sqrt(2)-1)*min(dx, dy)

    if not free(*start) or not free(*goal):
        return []

    queue = [(heuristic(*start), 0.0, start)]
    costs, parents = {start: 0.0}, {}

    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost > costs[current]:
            continue

        if current == goal:
            route = [current]
            while current in parents:
                current = parents[current]
                route.append(current)
            return route[::-1]

        x, y = current
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if not free(nx, ny):
                continue
            if dx and dy and (not free(x+dx, y) or not free(x, y+dy)):
                continue

            nxt = (nx, ny)
            new_cost = cost + math.hypot(dx, dy)
            if new_cost >= costs.get(nxt, math.inf):
                continue

            costs[nxt], parents[nxt] = new_cost, current
            heapq.heappush(queue, (new_cost+heuristic(nx, ny), new_cost, nxt))

    return []

def line_clear(blocked, start, end):
    """Check every cell touched by a center-to-center segment, including corners."""
    h, w = blocked.shape

    def free(x, y):
        return 0 <= x < w and 0 <= y < h and not blocked[y, x]

    x, y = start
    dx, dy = end[0] - x, end[1] - y
    nx, ny = abs(dx), abs(dy)
    sx, sy = int(np.sign(dx)), int(np.sign(dy))
    ix = iy = 0
    if not free(x, y):
        return False
    while ix < nx or iy < ny:
        crossing = (2 * ix + 1) * ny - (2 * iy + 1) * nx
        if crossing == 0:
            # A corner touches both adjacent cells; neither may be blocked.
            if not free(x + sx, y) or not free(x, y + sy):
                return False
            x, y = x + sx, y + sy
            ix, iy = ix + 1, iy + 1
        elif crossing < 0:
            x, ix = x + sx, ix + 1
        else:
            y, iy = y + sy, iy + 1
        if not free(x, y):
            return False
    return True


def shorten_path(blocked, cells):
    """Greedily connect to the farthest visible route point on the inflated grid."""
    if not cells:
        return []
    result = [cells[0]]
    index = 0
    while index < len(cells) - 1:
        next_index = len(cells) - 1
        while next_index > index + 1 and not line_clear(
                blocked, cells[index], cells[next_index]):
            next_index -= 1
        result.append(cells[next_index])
        index = next_index
    return result


class AStarNavigation(Node):
    def __init__(self):
        super().__init__('astar_navigation')

        # Flight configuration
        self.altitude = 1.5
        self.tolerance = 0.10
        self.goal = np.array([1.0, 3.5])
        self.clearance = 0.35
        self.map_file = str(FilePath(map_directory()) / 'arena_latest.npz')

        # Sensors and flight state
        self.position, self.scan = None, None
        self.yaw, self.hold_yaw = 0.0, 0.0
        self.last_odom = self.last_scan = 0.0
        self.look_since = None
        self.state = 'TAKEOFF'

        # Load map and prepare A*
        self.saved_map = load_map(self.map_file)
        m = self.saved_map
        q, p = m.info.origin.orientation, m.info.origin.position
        self.origin = np.array([p.x, p.y, math.atan2(2*q.w*q.z, 1-2*q.z*q.z)])
        self.resolution = m.info.resolution
        data = np.asarray(m.data).reshape(m.info.height, m.info.width)
        self.blocked = blocked_grid(data, self.resolution, self.clearance)

        # Route planning
        self.route = []
        self.route_index = 0
        self.last_plan = -math.inf

        # ROS interfaces
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', latched)
        self.path_pub = self.create_publisher(Path, '/astar/path', latched)

        self.saved_map.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.saved_map)

        self.control_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.05, self.control, clock=self.control_clock)


    def odom_cb(self, msg):
        """Use Gazebo odometry directly, without adding the spawn position twice."""
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.position = np.array([p.x, p.y, p.z])
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        self.last_odom = time.monotonic()


    def sector(self, start, end):
        """Return nearest metres; positive infinity is clear, invalid rays are unknown."""
        msg = self.scan
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


    def send(self, vx=0., vy=0., wz=0., altitude=True):
        """Publish body velocities, maintaining altitude unless odometry is stale."""
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        if altitude and self.position is not None:
            msg.linear.z = float(np.clip(.7*(self.altitude-self.position[2]), -.25, .25))
        self.pub_cmd.publish(msg)


    def scan_cb(self, msg):
        """Use raw scans only; this mode does not run SLAM."""
        self.scan, self.last_scan = msg, time.monotonic()


    def plan(self):
        start = grid_cell(self.position[:2], self.origin, self.resolution)
        goal = grid_cell(self.goal, self.origin, self.resolution)
        cells = astar(self.blocked, start, goal)

        if not cells:
            self.get_logger().warning('No safe A* route; hovering')
            return

        bends = shorten_path(self.blocked, cells)

        angle = self.origin[2]
        cs, sn = math.cos(angle), math.sin(angle)
        self.route = []

        for x, y in bends:
            x, y = (x + 0.5) * self.resolution, (y + 0.5) * self.resolution
            self.route.append(self.origin[:2] + [cs*x - sn*y, sn*x + cs*y])

        self.route.append(self.goal)
        self.route_index = 0

        path = Path()
        path.header = self.saved_map.header
        path.header.stamp = self.get_clock().now().to_msg()

        for x, y in self.route:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.path_pub.publish(path)



    def hold_position(self):
        delta = self.goal - self.position[:2]
        dist = np.linalg.norm(delta)
        v = delta * min(0.6, 0.1 / dist) if dist > 0.02 else np.zeros(2)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        wz = np.clip(wrap(self.hold_yaw - self.yaw), -0.2, 0.2)
        self.send(c*v[0] + s*v[1], -s*v[0] + c*v[1], wz)



    def control(self):
        now = time.monotonic()

        if self.position is None or now - self.last_odom > 0.6:
            self.send(altitude=False)
            return

        if self.state == 'TAKEOFF':
            self.send()
            if abs(self.position[2] - self.altitude) < 0.1:
                self.state = 'NAVIGATE'
            return

        distance = np.linalg.norm(self.goal - self.position[:2])
        if self.state == 'HOLD' or distance < self.tolerance:
            if self.state != 'HOLD':
                self.hold_yaw = self.yaw
                self.state = 'HOLD'
            self.hold_position()
            return

        if not self.route:
            self.send()
            if now - self.last_plan > 2.0:
                self.last_plan = now
                self.plan()
            return

        while self.route_index < len(self.route) - 1 and np.linalg.norm(self.route[self.route_index] - self.position[:2]) < 0.06:
            self.route_index += 1

        delta = self.route[self.route_index] - self.position[:2]
        distance = np.linalg.norm(delta)
        error = wrap(math.atan2(delta[1], delta[0]) - self.yaw)

        if abs(error) > 0.08:
            self.look_since = None
            self.send(wz=np.clip(1.3 * error, -0.4, 0.4))
            return

        if self.look_since is None:
            self.look_since = now

        front = self.sector(-25, 25)
        if now - self.last_scan > 0.6 or front is None or front <= 0.35 or now - self.look_since < 0.2 or self.last_scan <= self.look_since:
            self.send()
            return

        self.send(min(0.18, 0.6 * distance), wz=np.clip(error, -0.15, 0.15))



def main(args=None):
    """Run saved-map navigation."""
    rclpy.init(args=args)
    node = AStarNavigation()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
