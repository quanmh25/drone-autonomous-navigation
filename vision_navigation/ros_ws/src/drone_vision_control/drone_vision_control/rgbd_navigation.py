"""Reactive waypoint navigation with front depth and right-wall following."""

from functools import partial
import math
from pathlib import Path as FilePath
import time

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener

from drone_vision_control.module_rgbd.depth_processing import depth_distance, depth_in_metres, right_wall_geometry
from drone_vision_control.module_rgbd.geometry import clamp, goal_geometry, on_goal_line, wrap, yaw_of
from drone_vision_control.module_rgbd.map_snapshot import default_map_directory, save_odom_snapshot



class RGBDNavigation(Node):
    def __init__(self):
        super().__init__('rgbd_navigation')

        # Navigation settings (edit directly here).
        self.p = {
            # Flight and waypoint arrival
            'altitude': 1.5, 'tolerance': 0.05, 'goal_speed': 0.30, 'wall_speed': 0.15, 'yaw_speed': 0.55,
            # Obstacle clearance and right-wall following
            'safe_distance': 0.50, 'front_limit': 0.30,
        }

        # Camera images and odometry (filled by callbacks).
        self.bridge = CvBridge()
        self.position, self.depth, self.right_depth = None, None, None
        self.yaw, self.depth_count = 0.0, 0
        self.last_odom = self.last_depth = self.last_right_depth = 0.0

        # Takeoff, waypoint progress and final position hold.
        self.state, self.index = 'TAKEOFF', 0
        self.waypoints = [(0.0, 0.0)]
        self.leg_start, self.hold_xy, self.hold_yaw = None, None, None

        # Wall-following progress and timeout checks.
        self.hit_goal_distance = None
        self.follow_heading = 0.0
        self.wall_started, self.wall_progress_time, self.wall_progress_position = None, None, None
        self.next_goal_check, self.turning_corner = 0.0, False

        # Wait for a fresh front image after changing heading.
        self.look_since, self.goal_scan_heading = None, None
        self.look_frame = 0

        # Save the completed map while holding the goal.
        share = get_package_share_directory('drone_vision_control')
        self.save_map_path = str(FilePath(default_map_directory(share)) / 'arena_latest.npz')
        self.saved_grid, self.done_since = None, None
        self.snapshot_saved, self.last_save_attempt = False, -math.inf
        self.snapshot_tf = Buffer()
        self.snapshot_listener = TransformListener(self.snapshot_tf, self)

        # Input/output topics and control loop (20 Hz).
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/front_rgbd/depth_image', self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/right_rgbd/depth_image', partial(self.depth_cb, right=True),
            qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/rtabmap/map', self.map_cb, 10)
        # Keep the watchdog alive even before /clock arrives or when Gazebo pauses.
        self.control_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.05, self.control, clock=self.control_clock)

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.position = np.array([p.x, p.y, p.z], dtype=float)
        self.yaw = yaw_of(q)
        self.last_odom = time.monotonic()


    def depth_cb(self, msg, right=False):
        """Read either depth camera in metres and track its freshness."""
        if msg.encoding not in ('16UC1', '32FC1'):
            return
        try:
            depth = depth_in_metres(msg, self.bridge)
            if right:
                self.right_depth = depth
                self.last_right_depth = time.monotonic()
            else:
                self.depth = depth
                self.depth_count += 1
                self.last_depth = time.monotonic()
        except Exception as error:
            camera = 'Right depth' if right else 'Depth'
            self.get_logger().warning(f'{camera} error: {error}', throttle_duration_sec=2.)

    def map_cb(self, msg):
        self.saved_grid = msg


    def send(self, vx=0.0, vy=0.0, vz=None, wz=0.0):
        # Navigation uses odom/world XY; Gazebo VelocityControl expects body XY.
        msg = Twist()
        msg.linear.x = float(math.cos(self.yaw) * vx + math.sin(self.yaw) * vy)
        msg.linear.y = float(-math.sin(self.yaw) * vx + math.cos(self.yaw) * vy)
        if vz is None:
            vz = (clamp(0.7 * (self.p['altitude'] - self.position[2]), 0.15) if self.position is not None else 0.0)
        msg.linear.z = float(vz)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def set_state(self, state):
        if self.state != state:
            self.state = state
            self.look_since = None
            self.goal_scan_heading = None


    def rotate(self, target):
        error = wrap(target - self.yaw)
        self.send(wz=clamp(1.3 * error, self.p['yaw_speed']))
        return abs(error) < 0.07

    def look(self, heading, hold=True):
        error = wrap(heading - self.yaw)

        if abs(error) >= 0.07:
            self.look_since = None
            self.rotate(heading)
            return False, None

        if hold:
            self.send()

        if self.look_since is None:
            self.look_since = time.monotonic()
            self.look_frame = self.depth_count
            return False, None

        if time.monotonic() - self.look_since < 0.20 or self.depth_count <= self.look_frame:
            return False, None

        return True, depth_distance(self.depth, self.p['front_limit'])

    def goal_clearance(self, hysteresis=0.0):
        # Allow a reachable goal in front of a wall, while preserving a minimum
        # stopping clearance. Camera offset and vehicle size fit in this margin.
        distance, _ = goal_geometry(self.position[:2], self.waypoints[self.index])
        return max(self.p['front_limit'], min(
            self.p['safe_distance'] + hysteresis, distance + self.p['tolerance']))

    def wall_follow_stalled(self, now):
        if self.wall_started is None:
            return False
        if np.linalg.norm(self.position[:2] - self.wall_progress_position) >= 0.15:
            self.wall_progress_position = self.position[:2].copy()
            self.wall_progress_time = now
        return now - self.wall_progress_time > 90.0 or now - self.wall_started > 600.0


    def hold_position(self):
        """Hold the final odom XY target and heading after reaching it."""
        error = self.hold_xy - self.position[:2]
        distance = float(np.linalg.norm(error))
        velocity = np.zeros(2)
        if distance > 0.02:
            velocity = error * min(0.6, 0.12 / distance)
        yaw_error = wrap(self.hold_yaw - self.yaw)
        self.send(velocity[0], velocity[1], wz=clamp(0.8 * yaw_error, 0.15))


    def finish_mapping(self, now):
        """Save once arrival settles, then keep holding the goal."""
        if not self.save_map_path or self.snapshot_saved:
            return
        if np.linalg.norm(self.hold_xy - self.position[:2]) > self.p['tolerance']:
            self.done_since = None
            return
        if self.done_since is None:
            self.done_since = now
        if now - self.done_since < 2. or now - self.last_save_attempt < 2.:
            return
        self.last_save_attempt = now
        if self.saved_grid is None:
            self.get_logger().warning(
                'Waiting for the first occupancy map before saving', throttle_duration_sec=5.)
            return
        try:
            transform = None
            if self.saved_grid.header.frame_id != 'odom':
                transform = self.snapshot_tf.lookup_transform(
                    'odom', self.saved_grid.header.frame_id, Time()).transform
            filename = save_odom_snapshot(self.saved_grid, self.save_map_path, transform)
        except (TransformException, OSError, ValueError) as error:
            self.get_logger().warning(f'Map save failed; holding position: {error}')
            return
        self.snapshot_saved = True
        self.get_logger().info(f'Map saved: {filename}')

    def control(self):
        now = time.monotonic()

        if self.position is None or now - self.last_odom > 0.6:
            self.send(vz=0.0)
            self.get_logger().warning(
                'No fresh /odom; holding position', throttle_duration_sec=3.)
            return

        if self.state == 'DONE':
            self.hold_position()
            self.finish_mapping(now)
            return

        if self.state == 'BLOCKED':
            self.send()
            return

        if self.state == 'TAKEOFF':
            dz = self.p['altitude'] - self.position[2]
            if abs(dz) < 0.12:
                self.leg_start = self.position[:2].copy()
                self.set_state('GOAL')
                self.send()
            else:
                self.send(vz=clamp(0.7*dz, 0.25))
            return

        wall_states = ('TURN_LEFT', 'LOOK_GOAL', 'WALL_FOLLOW')
        if self.state in wall_states and self.wall_follow_stalled(now):
            self.send()
            self.get_logger().warning('Wall following timed out or made no progress')
            self.set_state('BLOCKED')
            return

        if self.depth is None or now - self.last_depth > 0.6:
            self.send()
            self.get_logger().warning('Waiting for RGB-D depth', throttle_duration_sec=3.0)
            return

        if self.state == 'GOAL':
            self.go_to_goal()
        elif self.state == 'TURN_LEFT':
            if self.rotate(self.follow_heading):
                self.set_state('WALL_FOLLOW')
        elif self.state == 'WALL_FOLLOW':
            self.follow_wall()
        elif self.state == 'LOOK_GOAL':
            self.look_goal()


    def go_to_goal(self):
        distance_xy, heading = goal_geometry(self.position[:2], self.waypoints[self.index])

        if distance_xy < self.p['tolerance']:
            self.index += 1
            self.send()

            if self.index >= len(self.waypoints):
                self.hold_xy = np.array(self.waypoints[-1], dtype=float)
                self.hold_yaw = self.yaw
                self.set_state('DONE')
                self.hold_position()
            else:
                self.leg_start = self.position[:2].copy()
            return

        yaw_error = wrap(heading - self.yaw)
        if (self.goal_scan_heading is None or abs(wrap(heading - self.goal_scan_heading)) > 0.07):
            self.goal_scan_heading = heading
            self.look_since = None

        ready, front = self.look(self.goal_scan_heading, hold=False)
        if not ready:
            if self.look_since is not None:
                self.send()
            return

        if front is None:
            self.send()
            return

        if front <= self.goal_clearance():
            self.send()
            self.wall_started = time.monotonic()
            self.wall_progress_time = self.wall_started
            self.wall_progress_position = self.position[:2].copy()
            self.hit_goal_distance = distance_xy
            self.follow_heading = wrap(heading + math.pi / 2)
            self.turning_corner = False
            self.next_goal_check = self.wall_started + 5.0
            self.set_state('TURN_LEFT')
            return

        speed = min(self.p['goal_speed'], 0.6 * distance_xy)
        if front < 1.8:
            speed = min(speed, 0.12)

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        self.send(vx, vy, wz=clamp(yaw_error, 0.25))


    def follow_wall(self):
        now = time.monotonic()
        if self.right_depth is None or now - self.last_right_depth > 0.6:
            self.send()
            self.get_logger().warning('Waiting for right depth', throttle_duration_sec=3.)
            return
        front = depth_distance(self.depth, self.p['front_limit'])
        right = depth_distance(self.right_depth, self.p['front_limit'])
        if front is None or right is None:
            self.send()
            return
        distance, _ = goal_geometry(self.position[:2], self.waypoints[self.index])
        passed_end = (right > self.p['safe_distance'] + 0.40 and distance < self.hit_goal_distance - 0.35)
        if (now >= self.next_goal_check and
                (on_goal_line(
                    self.position[:2], self.leg_start, self.waypoints[self.index],
                    self.hit_goal_distance) or passed_end)):
            self.follow_heading = self.yaw
            self.send()
            self.set_state('LOOK_GOAL')
            return
        if front < self.p['front_limit']:
            self.turning_corner = True
        elif front > self.p['front_limit'] + 0.15:
            self.turning_corner = False
        if self.turning_corner:
            self.send(wz=0.35)
            return
        geometry = right_wall_geometry(self.right_depth)
        if geometry is not None:
            perpendicular, tangent = geometry
            error = self.p['safe_distance'] - perpendicular
            if abs(error) < 0.06:
                error = 0.0
            # Heading feedback damps the weave caused by distance-only control.
            wz = clamp(1.2 * tangent + 0.5 * error, 0.25)
        else:
            # Cautiously round an open wall end; do not turn sharply to chase it.
            error = self.p['safe_distance'] - right
            wz = 0.0 if abs(error) < 0.12 else clamp(0.3 * error, 0.18)
        speed = self.p['wall_speed'] if front > self.p['safe_distance'] + 0.15 else 0.07
        self.send(speed * math.cos(self.yaw), speed * math.sin(self.yaw), wz=wz)


    def look_goal(self):
        if self.goal_scan_heading is None:
            self.goal_scan_heading = goal_geometry(self.position[:2], self.waypoints[self.index])[1]
        ready, front = self.look(self.goal_scan_heading)

        if not ready:
            return

        if front is None:
            self.look_since = None
            return

        required_clearance = self.goal_clearance(hysteresis=0.15)
        if front > required_clearance:
            self.set_state('GOAL')
        else:
            self.next_goal_check = time.monotonic() + 10.0
            self.set_state('TURN_LEFT')


def main(args=None):
    rclpy.init(args=args)
    node = RGBDNavigation()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
