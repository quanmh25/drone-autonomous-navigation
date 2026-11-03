import math
from pathlib import Path
import time


from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


from drone_lidar_control.module_lidar.lidar_processing import sector
from drone_lidar_control.module_lidar.map_snapshot import map_directory, save_map


def wrap(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class LidarNavigation(Node):
    """Take off, seek each waypoint, follow left walls and hold on arrival."""
    def __init__(self):
        super().__init__('lidar_navigation')

        # Flight configuration
        self.waypoints = np.array([[2.0, 3.0]])
        self.altitude = 1.5
        self.speed, self.wall_speed = 0.25, 0.12
        self.wall_distance = 0.50
        self.stop_distance, self.clear_distance = 0.35, 0.55
        self.tolerance = 0.10

        # Map saving
        self.save_map_path = str(Path(map_directory()) / 'arena_latest.npz')
        self.latest_map, self.settled_since = None, None
        self.map_saved, self.last_save_attempt = False, -math.inf
        self.map_tf = Buffer()
        self.map_listener = TransformListener(self.map_tf, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, latched)

        # Sensors and navigation
        self.position, self.scan = None, None
        self.yaw, self.hold_yaw = 0.0, 0.0
        self.last_odom = self.last_scan = 0.0
        self.state, self.index = 'TAKEOFF', 0

        # Wall following
        self.hit_distance = math.inf
        self.wall_started = self.progress_time = 0.0
        self.progress_xy = None
        self.wall_heading = self.next_check = 0.0
        self.look_since = None

        # ROS interfaces
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.slam_scan_pub = self.create_publisher(LaserScan, '/slam/scan', qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.control_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.05, self.control, clock=self.control_clock)


    def odom_cb(self, msg):
        """Use Gazebo odometry directly, without adding the spawn position twice."""
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.position = np.array([p.x, p.y, p.z])
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        self.last_odom = time.monotonic()


    def scan_cb(self, msg):
        """Cache the scan and its reception time for the watchdog."""
        self.scan = msg
        self.last_scan = time.monotonic()
        # Build a single horizontal slice only after takeoff. This output has
        # no feedback into flight control, even if SLAM stops or corrects map TF.
        if (self.state != 'TAKEOFF' and self.position is not None and
                self.last_scan-self.last_odom <= .6 and
                abs(self.position[2]-self.altitude) < .1):
            self.slam_scan_pub.publish(msg)

    def map_cb(self, msg):
        """Observe SLAM output for saving only; never use it for navigation."""
        self.latest_map = msg

    def save_finished_map(self, now):
        """Save a stable final map, retrying failures without interrupting hover."""
        if not self.save_map_path or self.map_saved:
            return
        if np.linalg.norm(self.waypoints[-1]-self.position[:2]) > self.tolerance:
            self.settled_since = None
            return
        if self.settled_since is None:
            self.settled_since = now
        if now-self.settled_since < 3. or now-self.last_save_attempt < 2.:
            return
        self.last_save_attempt = now
        if self.latest_map is None:
            self.get_logger().warning('Waiting for /map before saving', throttle_duration_sec=5.)
            return
        try:
            transform = self.map_tf.lookup_transform(
                'odom', self.latest_map.header.frame_id, Time())
            path = save_map(self.latest_map, self.save_map_path, transform.transform)
        except (TransformException, ValueError, OSError) as error:
            self.get_logger().warning(f'Map save failed; hovering: {error}')
            return
        self.map_saved = True
        self.get_logger().info(f'Map saved: {path}')


    def send(self, vx=0., vy=0., wz=0., altitude=True):
        """Publish body velocities, maintaining altitude unless odometry is stale."""
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        if altitude and self.position is not None:
            msg.linear.z = float(np.clip(.7*(self.altitude-self.position[2]), -.25, .25))
        self.pub_cmd.publish(msg)

    def set_state(self, state):
        """Restart the fresh-scan wait when changing flight mode."""
        if state != self.state:
            self.state = state
            self.look_since = None


    def hold_position(self, now):
        delta = self.waypoints[-1] - self.position[:2]
        dist = np.linalg.norm(delta)
        v = delta * min(0.6, 0.1 / dist) if dist > 0.02 else np.zeros(2)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        wz = np.clip(wrap(self.hold_yaw - self.yaw), -0.2, 0.2)
        self.send(c*v[0] + s*v[1], -s*v[0] + c*v[1], wz)
        self.save_finished_map(now)


    def navigate(self, now, distance, heading, front):
        target = self.wall_heading if self.state == 'RETURN_WALL' else heading
        turn = wrap(target - self.yaw)

        if abs(turn) > 0.08:
            self.look_since = None
            self.send(wz=np.clip(1.3 * turn, -0.5, 0.5))
            return

        if self.look_since is None:
            self.look_since = now
        if now - self.look_since < 0.2 or self.last_scan <= self.look_since:
            self.send()
            return

        if self.state == 'RETURN_WALL':
            self.set_state('WALL')
            self.send()
            return

        required = max(self.stop_distance, min(self.clear_distance, distance + self.tolerance))
        if front <= required:
            self.send()
            if self.state == 'LOOK_GOAL':
                self.next_check = now + 5.0
                self.set_state('RETURN_WALL')
            else:
                self.hit_distance = distance
                self.wall_started = self.progress_time = now
                self.progress_xy = self.position[:2].copy()
                self.next_check = now + 5.0
                self.set_state('WALL')
            return

        if self.state == 'LOOK_GOAL':
            self.set_state('GOAL')
        self.send(min(self.speed, 0.6 * distance), wz=np.clip(turn, -0.2, 0.2))


    def follow_wall(self, now, distance, front, left):
        if now >= self.next_check and distance < self.hit_distance - 0.35:
            self.wall_heading = self.yaw
            self.set_state('LOOK_GOAL')
            self.send()
        elif front < self.clear_distance:
            self.send(wz=-0.4)
        else:
            turn = 0.22 if math.isinf(left) else np.clip(0.8 * (left - self.wall_distance), -0.3, 0.3)
            speed = 0.0 if left < self.stop_distance else self.wall_speed
            self.send(speed, wz=turn)


    def control(self):
        now = time.monotonic()

        if self.position is None or not np.isfinite(self.position).all() or not math.isfinite(self.yaw) or now - self.last_odom > 0.6:
            self.send(altitude=False)
            return

        if self.state == 'TAKEOFF':
            self.send()
            if abs(self.position[2] - self.altitude) < 0.1:
                self.set_state('GOAL')
            return

        if self.state == 'BLOCKED':
            self.send()
            return

        if self.state == 'HOLD':
            self.hold_position(now)
            return

        if self.scan is None or now - self.last_scan > 0.6:
            self.send()
            return

        front, left = sector(self.scan, -20, 20), sector(self.scan, 70, 100)
        if front is None or left is None:
            self.send()
            return

        delta = self.waypoints[self.index] - self.position[:2]
        distance = float(np.linalg.norm(delta))

        if distance < self.tolerance:
            self.send()
            if self.index == len(self.waypoints) - 1:
                self.hold_yaw = self.yaw
                self.set_state('HOLD')
            else:
                self.index += 1
                self.set_state('GOAL')
            return

        if self.state in ('WALL', 'LOOK_GOAL', 'RETURN_WALL'):
            if np.linalg.norm(self.position[:2] - self.progress_xy) >= 0.15:
                self.progress_xy = self.position[:2].copy()
                self.progress_time = now
            if now - self.progress_time > 90 or now - self.wall_started > 600:
                self.send()
                self.set_state('BLOCKED')
                return

        heading = math.atan2(delta[1], delta[0])
        if self.state in ('GOAL', 'LOOK_GOAL', 'RETURN_WALL'):
            self.navigate(now, distance, heading, front)
        elif self.state == 'WALL':
            self.follow_wall(now, distance, front, left)



def main(args=None):
    rclpy.init(args=args)
    node = LidarNavigation()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
