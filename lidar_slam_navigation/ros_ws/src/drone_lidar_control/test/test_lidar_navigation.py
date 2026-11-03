"""Verify LiDAR flight decisions without a running simulator."""

import math
from unittest.mock import Mock, patch

from drone_lidar_control.lidar_navigation import LidarNavigation
from drone_lidar_control.module_lidar.lidar_processing import sector
from nav_msgs.msg import Odometry
import numpy as np
import pytest
from sensor_msgs.msg import LaserScan


@pytest.fixture
def node():
    with patch('rclpy.node.Node.__init__', return_value=None), \
            patch('drone_lidar_control.lidar_navigation.map_directory',
                  return_value='/tmp/lidar_test_maps'), \
            patch('drone_lidar_control.lidar_navigation.Buffer'), \
            patch('drone_lidar_control.lidar_navigation.TransformListener'), \
            patch.object(LidarNavigation, 'create_publisher'), \
            patch.object(LidarNavigation, 'create_subscription'), \
            patch.object(LidarNavigation, 'create_timer'), \
            patch.object(LidarNavigation, 'get_logger', return_value=Mock()):
        nav = LidarNavigation()
    nav.get_logger = Mock(return_value=Mock())
    nav.position = np.array([0., 0., 1.5])
    nav.waypoints = np.array([[3., 0.]])
    nav.last_odom = nav.last_scan = 10.
    nav.state = 'GOAL'
    scan = LaserScan()
    scan.angle_min, scan.angle_increment = -1.85, 3.7/639
    scan.range_min, scan.range_max = .08, 10.
    scan.ranges = [math.inf]*640
    nav.scan_cb(scan)
    nav.last_scan = 10.
    return nav


def tick(node, now=10.):
    with patch('drone_lidar_control.lidar_navigation.time.monotonic', return_value=now):
        node.control()
    return node.pub_cmd.publish.call_args.args[0]


def test_stale_odometry_stops_all_axes(node):
    msg = tick(node, 11.)
    assert msg.linear.x == msg.linear.y == msg.linear.z == msg.angular.z == 0.


def test_stale_scan_holds_altitude(node):
    node.last_scan = 8.
    node.position[2] = 1.3
    msg = tick(node)
    assert msg.linear.x == msg.angular.z == 0.
    assert msg.linear.z > 0.


def test_invalid_scan_is_unknown_and_stops(node):
    node.scan.ranges = [math.nan]*640
    assert sector(node.scan, -20, 20) is None
    assert tick(node).linear.x == 0.


def test_infinite_scan_is_clear_but_close_returns_are_preserved(node):
    assert math.isinf(sector(node.scan, -20, 20))
    node.scan.ranges = [.03]*640
    assert sector(node.scan, -20, 20) == pytest.approx(.03)


def test_goal_motion_waits_for_a_new_scan(node):
    assert tick(node).linear.x == 0.
    assert tick(node, 10.3).linear.x == 0.
    node.last_scan = 10.3
    assert tick(node, 10.3).linear.x > 0.


def test_front_wall_enters_wall_mode_and_turns_right(node):
    node.scan.ranges = [.3]*640
    tick(node)
    node.last_scan = 10.3
    tick(node, 10.3)
    assert node.state == 'WALL'
    msg = tick(node, 10.3)
    assert msg.linear.x == 0.
    assert msg.angular.z < 0.


def test_left_wall_feedback_turns_away_from_close_wall(node):
    node.state = 'WALL'
    node.progress_xy = node.position[:2].copy()
    node.progress_time = node.wall_started = 10.
    node.next_check = 20.
    with patch('drone_lidar_control.lidar_navigation.sector', side_effect=[2., .4]):
        assert tick(node).angular.z < 0.
    with patch('drone_lidar_control.lidar_navigation.sector', side_effect=[2., .7]):
        assert tick(node).angular.z > 0.


def test_wall_timeout_hovers(node):
    node.state = 'WALL'
    node.progress_xy = node.position[:2].copy()
    node.progress_time = node.wall_started = -100.
    assert tick(node).linear.x == 0.
    assert node.state == 'BLOCKED'


def test_arrival_advances_waypoints_then_holds(node):
    node.waypoints = np.array([[0., 0.], [1., 0.]])
    tick(node)
    assert node.index == 1 and node.state == 'GOAL'
    node.position[0] = 1.
    tick(node)
    assert node.state == 'HOLD'
    assert tick(node).linear.x == 0.


def test_yaw_uses_quaternion_and_does_not_add_spawn_offset(node):
    msg = Odometry()
    msg.pose.pose.position.x = 1.
    msg.pose.pose.orientation.z = math.sin(math.pi/4)
    msg.pose.pose.orientation.w = math.cos(math.pi/4)
    node.odom_cb(msg)
    assert node.yaw == pytest.approx(math.pi/2)
    assert node.position[0] == 1.


def test_wall_only_checks_goal_after_progress(node):
    node.state = 'WALL'
    node.progress_xy = node.position[:2].copy()
    node.progress_time = node.wall_started = 10.
    node.hit_distance = 3.
    tick(node)
    assert node.state == 'WALL'
    node.position[0] = .5
    tick(node)
    assert node.state == 'LOOK_GOAL'


def test_blocked_goal_check_returns_to_previous_wall_heading(node):
    node.state = 'LOOK_GOAL'
    node.progress_xy = node.position[:2].copy()
    node.progress_time = node.wall_started = 10.
    node.wall_heading = math.pi/2
    node.scan.ranges = [.3]*640
    tick(node)
    node.last_scan = 10.3
    tick(node, 10.3)
    assert node.state == 'RETURN_WALL'
    assert tick(node, 10.3).angular.z > 0.


def test_wall_uses_left_scan_and_curves_left_at_open_end(node):
    node.state = 'WALL'
    node.progress_xy = node.position[:2].copy()
    node.progress_time = node.wall_started = 10.
    node.next_check = 20.
    with patch('drone_lidar_control.lidar_navigation.sector',
               side_effect=[2., math.inf]) as measure:
        msg = tick(node)
    assert measure.call_args_list[1].args == (node.scan, 70, 100)
    assert msg.linear.x > 0.
    assert msg.angular.z > 0.


def test_mapping_starts_at_flight_altitude_and_preserves_raw_scan(node):
    node.slam_scan_pub.reset_mock()
    node.state = 'TAKEOFF'
    node.scan_cb(node.scan)
    node.slam_scan_pub.publish.assert_not_called()
    node.state = 'GOAL'
    node.position[2] = .8
    node.scan_cb(node.scan)
    node.slam_scan_pub.publish.assert_not_called()
    node.position[2] = 1.5
    with patch('drone_lidar_control.lidar_navigation.time.monotonic', return_value=10.):
        node.scan_cb(node.scan)
    node.slam_scan_pub.publish.assert_called_once_with(node.scan)


def test_navigation_does_not_wait_for_mapping(node):
    # No map data or map TF exists in this fixture; a scan and odometry suffice.
    node.slam_scan_pub = Mock()
    tick(node)
    node.last_scan = 10.3
    assert tick(node, 10.3).linear.x > 0.


def test_save_waits_for_settling_and_retries_without_affecting_flight(node, tmp_path):
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid

    node.save_map_path = str(tmp_path/'saved.npz')
    node.position[:2] = node.waypoints[-1]
    node.map_tf = Mock()
    tf = TransformStamped()
    tf.transform.rotation.w = 1.
    node.map_tf.lookup_transform.return_value = tf
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.width = msg.info.height = 2
    msg.info.resolution = .1
    msg.info.origin.orientation.w = 1.
    msg.data = [0, 0, -1, 100]
    node.latest_map = msg
    node.save_finished_map(10.)
    assert not node.map_saved
    with patch('drone_lidar_control.lidar_navigation.save_map', side_effect=OSError('busy')):
        node.save_finished_map(13.)
    assert not node.map_saved
    node.save_finished_map(15.)
    assert node.map_saved
    assert (tmp_path/'saved.npz').exists()
