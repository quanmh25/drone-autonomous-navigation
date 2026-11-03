"""Follow a shortest grid path to a map-frame goal using RTAB-Map occupancy."""

import heapq
import math
from pathlib import Path as FilePath
import time

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener


def default_map_directory(share_directory):
    """Prefer workspace source config with both copied and symlink installs."""
    share = FilePath(share_directory).resolve()
    for root in (FilePath.cwd(), *FilePath.cwd().parents, *share.parents):
        package = root / 'src' / 'drone_vision_control'
        if (package / 'package.xml').is_file() and (package / 'config').is_dir():
            return str(package / 'config')
    return str(share / 'config')


def inflated_grid(data, resolution, clearance, threshold=50):
    """Block unknown cells and inflate obstacles and map edges by clearance."""
    if resolution <= 0 or clearance < 0:
        raise ValueError('Invalid resolution or clearance')
    occupied = (data >= threshold) | (data < 0)
    blocked = occupied.copy()
    radius = math.ceil(clearance / resolution)
    h, w = data.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if math.hypot(dx, dy) * resolution > clearance + resolution * math.sqrt(2):
                continue
            sy, ey = max(0, -dy), min(h, h - dy)
            sx, ex = max(0, -dx), min(w, w - dx)
            if ey > sy and ex > sx:
                blocked[sy + dy:ey + dy, sx + dx:ex + dx] |= occupied[sy:ey, sx:ex]
    if radius:
        blocked[:radius] = blocked[-radius:] = True
        blocked[:, :radius] = blocked[:, -radius:] = True
    return blocked


def astar(blocked, start, goal):
    """Return the minimum-length grid path; forbid diagonal corner cutting."""
    h, w = blocked.shape

    def free(cell):
        x, y = cell
        return 0 <= x < w and 0 <= y < h and not blocked[y, x]

    def heuristic(cell):
        dx, dy = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    if not free(start) or not free(goal):
        return []
    queue = [(heuristic(start), 0., start)]
    costs, parents = {start: 0.}, {}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost > costs[current]:
            continue
        if current == goal:
            result = [current]
            while current in parents:
                current = parents[current]
                result.append(current)
            return result[::-1]
        x, y = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = x + dx, y + dy
            if not free(nxt):
                continue
            if dx and dy and (not free((x + dx, y)) or not free((x, y + dy))):
                continue
            candidate = cost + math.hypot(dx, dy)
            if candidate < costs.get(nxt, math.inf):
                costs[nxt], parents[nxt] = candidate, current
                heapq.heappush(queue, (candidate + heuristic(nxt), candidate, nxt))
    return []


def world_to_cell(x, y, origin, resolution):
    """Convert a world point using the full planar occupancy-grid origin pose."""
    ox, oy, yaw = origin
    dx, dy = x - ox, y - oy
    return (math.floor((math.cos(yaw) * dx + math.sin(yaw) * dy) / resolution),
            math.floor((-math.sin(yaw) * dx + math.cos(yaw) * dy) / resolution))


def cell_to_world(cell, origin, resolution):
    """Return a cell center in the world frame."""
    ox, oy, yaw = origin
    x, y = (cell[0] + 0.5) * resolution, (cell[1] + 0.5) * resolution
    return ox + math.cos(yaw) * x - math.sin(yaw) * y, \
        oy + math.sin(yaw) * x + math.cos(yaw) * y


def compress_path(cells):
    """Keep bends, preserving the grid path without arbitrary shortcuts."""
    if len(cells) < 3:
        return cells
    result = [cells[0]]
    for a, b, c in zip(cells, cells[1:], cells[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            result.append(b)
    return result + [cells[-1]]


def load_snapshot(filename):
    """Restore exact occupancy values and grid geometry without live SLAM."""
    with np.load(FilePath(filename).expanduser(), allow_pickle=False) as archive:
        data = archive['data']
        resolution = float(archive['resolution'])
        origin = archive['origin']
        frame = str(archive['frame_id'])
        if (data.ndim != 2 or not data.size or resolution <= 0 or
                len(origin) != 3 or not np.isfinite(origin).all() or not frame or
                np.any((data < -1) | (data > 100))):
            raise ValueError('Invalid saved grid')
        msg = OccupancyGrid()
        msg.header.frame_id = frame
        msg.info.resolution = resolution
        msg.info.height, msg.info.width = data.shape
        msg.info.origin.position.x, msg.info.origin.position.y = map(float, origin[:2])
        msg.info.origin.orientation.z = math.sin(float(origin[2]) / 2.)
        msg.info.origin.orientation.w = math.cos(float(origin[2]) / 2.)
        msg.data = data.astype(np.int8).ravel().tolist()
        return msg


def yaw_of(q):
    """Extract planar yaw from a ROS quaternion."""
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(angle):
    """Wrap an angular error."""
    return math.atan2(math.sin(angle), math.cos(angle))


class AStarNavigation(Node):
    """Plan only in observed free space and hover when a route is unavailable."""

    def __init__(self):
        super().__init__('astar_navigation')
        defaults = {'goal_x': 1.0, 'goal_y': 1.0, 'altitude': 1.5,
                    'clearance': 0.35, 'speed': 0.18, 'goal_tolerance': 0.10}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.config = {name: self.get_parameter(name).value for name in defaults}
        if (self.config['clearance'] < 0.2 or self.config['speed'] <= 0 or
                self.config['goal_tolerance'] <= 0):
            raise ValueError('Require clearance >=0.2 m and positive speed/tolerance')
        self.declare_parameter('map_file', '')
        self.map_file = self.get_parameter('map_file').value
        self.goal = np.array([self.config['goal_x'], self.config['goal_y']])
        self.position = None
        self.last_odom = self.last_depth = 0.
        self.front = None
        self.grid = self.map_msg = None
        self.origin = None
        self.signature = None
        self.route = []
        self.route_index = 0
        self.last_plan = -math.inf
        self.state = 'TAKEOFF'
        self.arrival_yaw = None
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, '/astar/path', latched)
        if self.map_file:
            saved_map = load_snapshot(self.map_file)
            saved_map.header.stamp = self.get_clock().now().to_msg()
            self.saved_map_pub = self.create_publisher(OccupancyGrid, '/astar/saved_map', latched)
            self.saved_map_pub.publish(saved_map)
            self.map_cb(saved_map)
        else:
            self.create_subscription(OccupancyGrid, '/rtabmap/map', self.map_cb, latched)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/front_rgbd/depth_image', self.depth_cb,
                                 qos_profile_sensor_data)
        self.control_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.05, self.control, clock=self.control_clock)
        self.get_logger().info(f'A* goal in map: {self.goal}; unknown cells are blocked')

    def odom_cb(self, msg):
        """Use odometry for altitude and a data-loss watchdog."""
        self.position = msg.pose.pose.position
        self.last_odom = time.monotonic()

    def depth_cb(self, msg):
        """Stop for unexpected obstacles ahead, even if the map is incomplete."""
        try:
            if msg.encoding not in ('16UC1', '32FC1'):
                return
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = np.asarray(depth, dtype=float)
            if msg.encoding == '16UC1':
                depth /= 1000.
            h, w = depth.shape
            roi = depth[int(h * .2):int(h * .55), int(w * .3):int(w * .7)]
            valid = roi[np.isfinite(roi) & (roi > 0)]
            self.front = (float(np.percentile(valid, 5)) if valid.size else
                          10. if np.count_nonzero(np.isposinf(roi)) > roi.size / 2 else None)
            self.last_depth = time.monotonic()
        except (CvBridgeError, ValueError, TypeError) as error:
            self.front = None
            self.get_logger().warning(str(error), throttle_duration_sec=3.)

    def map_cb(self, msg):
        """Invalidate paths as soon as map contents or geometry change."""
        if msg.info.resolution <= 0 or len(msg.data) != msg.info.width * msg.info.height:
            return
        if not msg.header.frame_id or not msg.data:
            return
        q = msg.info.origin.orientation
        origin = (msg.info.origin.position.x, msg.info.origin.position.y, yaw_of(q))
        signature = (msg.header.frame_id, msg.info.width, msg.info.height,
                     msg.info.resolution, origin, bytes(np.asarray(msg.data, dtype=np.int8)))
        if signature == self.signature:
            return
        self.send()  # Stop before potentially expensive inflation/replanning.
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.grid = inflated_grid(data, msg.info.resolution, self.config['clearance'])
        self.origin, self.map_msg, self.signature = origin, msg, signature
        self.route = []
        self.publish_path([])

    def send(self, vx=0., vy=0., wz=0., altitude=True):
        """Publish body-frame planar velocity and altitude feedback."""
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        if altitude and self.position is not None:
            msg.linear.z = float(np.clip(.7 * (self.config['altitude'] - self.position.z),
                                         -.25, .25))
        self.cmd_pub.publish(msg)

    def publish_path(self, points):
        """Expose the actual A* route in RViz."""
        msg = Path()
        msg.header.frame_id = self.map_msg.header.frame_id if self.map_msg else 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def plan(self, xy):
        """Plan the exact shortest path on the inflated eight-connected grid."""
        resolution = self.map_msg.info.resolution
        start = world_to_cell(*xy, self.origin, resolution)
        goal = world_to_cell(*self.goal, self.origin, resolution)
        cells = astar(self.grid, start, goal)
        if not cells:
            self.route = []
            self.publish_path([])
            self.get_logger().warning(
                'No free known route to goal (unknown/blocked/outside map); hovering',
                throttle_duration_sec=5.)
            return
        self.route = [cell_to_world(c, self.origin, resolution) for c in compress_path(cells)]
        # Follow exact requested coordinates after reaching the goal cell.
        self.route.append(tuple(self.goal))
        self.route_index = 0
        self.publish_path(self.route)

    def control(self):
        """Take off, await a valid map/TF, follow the route, then hold the goal."""
        now = time.monotonic()
        if self.position is None or now - self.last_odom > .6:
            self.send(altitude=False)
            return
        if self.state == 'TAKEOFF':
            self.send()
            if abs(self.position.z - self.config['altitude']) < .10:
                self.state = 'NAVIGATE'
            return
        if self.map_msg is None:
            self.send()
            self.get_logger().warning('Waiting for /rtabmap/map', throttle_duration_sec=5.)
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.map_msg.header.frame_id, 'base_link', Time())
            age = (self.get_clock().now().nanoseconds -
                   (tf.header.stamp.sec * 10**9 + tf.header.stamp.nanosec)) * 1e-9
            if age < -.1 or age > .6:
                self.send()
                return
        except TransformException:
            self.send()
            self.get_logger().warning('Waiting for map -> base_link TF', throttle_duration_sec=5.)
            return
        xy = np.array([tf.transform.translation.x, tf.transform.translation.y])
        yaw = yaw_of(tf.transform.rotation)
        error = self.goal - xy
        distance = float(np.linalg.norm(error))
        if self.state == 'HOLD' or distance < self.config['goal_tolerance']:
            if self.state != 'HOLD':
                self.state, self.arrival_yaw = 'HOLD', yaw
                self.get_logger().info(f'Reached map goal {self.goal}; holding position')
            velocity = error * min(.6, .10 / max(distance, 1e-6)) if distance > .02 else error * 0
            c, s = math.cos(yaw), math.sin(yaw)
            self.send(c * velocity[0] + s * velocity[1],
                      -s * velocity[0] + c * velocity[1],
                      float(np.clip(wrap(self.arrival_yaw - yaw), -.15, .15)))
            return
        if now - self.last_plan > 1. and not self.route:
            self.last_plan = now
            self.send()
            self.plan(xy)
            return  # Refresh TF/odom before moving after a potentially slow plan.
        if not self.route:
            self.send()
            return
        while (self.route_index < len(self.route) - 1 and
               np.linalg.norm(np.asarray(self.route[self.route_index]) - xy) < .08):
            self.route_index += 1
        target = np.asarray(self.route[self.route_index])
        delta = target - xy
        heading_error = wrap(math.atan2(delta[1], delta[0]) - yaw)
        if abs(heading_error) > .12:
            self.send(wz=float(np.clip(1.2 * heading_error, -.4, .4)))
            return
        if now - self.last_depth > .6 or self.front is None or self.front <= .3:
            self.send()
            self.route = []
            self.get_logger().warning('Depth stop; waiting for a safe map route',
                                      throttle_duration_sec=5.)
            return
        speed = min(self.config['speed'], .6 * float(np.linalg.norm(delta)))
        self.send(speed, wz=float(np.clip(heading_error, -.15, .15)))

    def destroy_node(self):
        """Stop commands before ROS shutdown when possible."""
        if self.context.ok():
            self.send(altitude=False)
        return super().destroy_node()


def main(args=None):
    """Run the A* navigation node."""
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
