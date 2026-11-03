"""Regression checks for navigation decisions without a running simulator."""

import math
from unittest.mock import Mock, patch

from drone_vision_control.module_rgbd.depth_processing import depth_distance, right_wall_geometry
from drone_vision_control.rgbd_navigation import RGBDNavigation
import numpy as np
import pytest


@pytest.fixture
def node():
    with patch('rclpy.node.Node.__init__', return_value=None), \
            patch('drone_vision_control.rgbd_navigation.get_package_share_directory',
                  return_value='/tmp/rgbd_test_share'), \
            patch('drone_vision_control.rgbd_navigation.Buffer'), \
            patch('drone_vision_control.rgbd_navigation.TransformListener'), \
            patch.object(RGBDNavigation, 'create_publisher'), \
            patch.object(RGBDNavigation, 'create_subscription'), \
            patch.object(RGBDNavigation, 'create_timer'), \
            patch.object(RGBDNavigation, 'get_logger', return_value=Mock()):
        nav = RGBDNavigation()
        nav.position = np.array([0., 0., 1.5])
        nav.depth = np.full((48, 64), 5., dtype=np.float32)
        yield nav


def enter_wall_follow(node):
    """Trigger avoidance through the same front obstacle decision used in flight."""
    node.waypoints = [(3., 0.)]
    node.state = 'GOAL'
    with patch.object(node, 'look', return_value=(True, 0.4)):
        node.go_to_goal()
    assert node.state == 'TURN_LEFT'


def test_world_velocity_is_converted_to_body(node):
    node.yaw = math.pi / 2
    node.send(vx=0., vy=0.3)
    msg = node.cmd_pub.publish.call_args.args[0]
    assert msg.linear.x == pytest.approx(0.3)
    assert msg.linear.y == pytest.approx(0., abs=1e-9)


def test_rotate_holds_altitude(node):
    node.position[2] = 1.3
    node.rotate(1.)
    assert node.cmd_pub.publish.call_args.args[0].linear.z > 0.


def test_goal_scan_is_dispatched(node):
    node.state = 'LOOK_GOAL'
    node.look_goal = Mock()
    with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=1.):
        node.last_odom = node.last_depth = 1.
        node.control()
    node.look_goal.assert_called_once()


def test_final_waypoint_does_not_index_past_list(node):
    node.state = 'GOAL'
    node.waypoints = [(0., 0.)]
    with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=3.):
        node.last_odom = node.last_depth = 3.
        node.control()
    assert node.state == 'DONE'


def test_new_heading_waits_for_fresh_depth(node):
    node.state = 'GOAL'
    node.waypoints = [(3., 0.)]
    node.go_to_goal()
    assert node.cmd_pub.publish.call_args.args[0].linear.x == 0.
    node.depth_count += 1
    node.look_since -= 0.3
    node.go_to_goal()
    assert node.cmd_pub.publish.call_args.args[0].linear.x > 0.


def test_close_depth_is_obstacle_and_nan_is_unknown(node):
    node.depth[:] = 0.1
    assert depth_distance(node.depth, node.p['front_limit']) == pytest.approx(0.1)
    node.depth[:] = np.nan
    assert depth_distance(node.depth, node.p['front_limit']) is None
    node.depth[:] = np.inf
    assert depth_distance(node.depth, node.p['front_limit']) == 10.


def test_goal_still_obstructed_keeps_following(node):
    node.waypoints = [(3., 0.)]
    node.state = 'LOOK_GOAL'
    node.look = Mock(return_value=(True, 0.5))
    node.look_goal()
    assert node.state == 'TURN_LEFT'


def test_control_timer_uses_steady_clock(node):
    from rclpy.clock import ClockType

    assert node.control_clock.clock_type == ClockType.STEADY_TIME
    assert node.create_timer.call_args.kwargs['clock'] is node.control_clock


def test_goal_before_wall_does_not_restart_avoidance(node):
    node.state = 'LOOK_GOAL'
    node.waypoints = [(0.5, 0.)]
    node.look = Mock(return_value=(True, 1.0))
    node.look_goal()
    assert node.state == 'GOAL'
    node.go_to_goal()
    assert node.state == 'GOAL'
    assert node.cmd_pub.publish.call_args.args[0].linear.x > 0.


def test_near_wall_keeps_minimum_clearance(node):
    node.state = 'GOAL'
    node.waypoints = [(0.5, 0.)]
    node.look = Mock(return_value=(True, 0.45))
    node.go_to_goal()
    assert node.state == 'TURN_LEFT'


def test_wall_scanning_without_progress_stops(node):
    enter_wall_follow(node)
    node.state = 'WALL_FOLLOW'
    with patch('drone_vision_control.rgbd_navigation.time.monotonic',
               return_value=node.wall_started + 91.):
        node.last_odom = node.wall_started + 91.
        node.control()
    assert node.state == 'BLOCKED'
    cmd = node.cmd_pub.publish.call_args.args[0]
    assert cmd.linear.x == 0. and cmd.angular.z == 0.


def test_wall_progress_refreshes_watchdog(node):
    enter_wall_follow(node)
    node.position[0] += 0.2
    assert not node.wall_follow_stalled(node.wall_started + 80.)
    assert not node.wall_follow_stalled(node.wall_started + 100.)
    assert node.wall_follow_stalled(node.wall_started + 171.)


def test_wall_follow_has_total_time_limit_even_with_progress(node):
    enter_wall_follow(node)
    node.position[0] += 0.2
    assert node.wall_follow_stalled(node.wall_started + 601.)


def test_side_depth_allows_forward_motion_without_scan(node):
    enter_wall_follow(node)
    node.state = 'WALL_FOLLOW'
    node.leg_start = node.position[:2].copy()
    node.right_depth = np.full((48, 64), node.p['safe_distance'])
    with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=10.):
        node.last_right_depth = 10.
        node.next_goal_check = 20.
        node.follow_wall()
    cmd = node.cmd_pub.publish.call_args.args[0]
    assert node.state == 'WALL_FOLLOW'
    assert cmd.linear.x > 0.
    assert cmd.angular.z == pytest.approx(0., abs=1e-9)


def test_missing_side_depth_stops_translation(node):
    enter_wall_follow(node)
    node.state = 'WALL_FOLLOW'
    node.follow_wall()
    assert node.cmd_pub.publish.call_args.args[0].linear.x == 0.


def side_wall_image(distance, tangent):
    """Project a plane at a known distance and angle into the right depth camera."""
    w = 160
    fx = w / (2. * math.tan(1.0472 / 2.))
    u = (np.arange(w) - (w - 1) / 2.) / fx
    slope = math.tan(tangent)
    offset = distance * math.hypot(1., slope)
    z = offset / (1. - slope * u)
    return np.tile(z, (120, 1))


@pytest.mark.parametrize('angle', [-0.3, 0., 0.3])
def test_estimates_side_wall_heading_and_perpendicular_distance(node, angle):
    node.right_depth = side_wall_image(0.8, angle)
    distance, tangent = right_wall_geometry(node.right_depth)
    assert distance == pytest.approx(0.8, abs=1e-6)
    assert tangent == pytest.approx(angle, abs=1e-6)


def test_heading_feedback_corrects_approach_even_at_target_distance(node):
    enter_wall_follow(node)
    node.leg_start = node.position[:2].copy()
    node.right_depth = side_wall_image(node.p['safe_distance'], 0.2)
    with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=10.):
        node.last_right_depth = 10.
        node.next_goal_check = 20.
        node.follow_wall()
    assert node.cmd_pub.publish.call_args.args[0].angular.z > 0.1


def test_corner_hysteresis_does_not_alternate_stop_and_forward(node):
    enter_wall_follow(node)
    node.leg_start = node.position[:2].copy()
    node.right_depth = side_wall_image(node.p['safe_distance'], 0.)
    with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=10.):
        node.last_right_depth = 10.
        node.next_goal_check = 20.
        node.depth[:] = 0.25
        node.follow_wall()
        node.depth[:] = 0.35
        node.follow_wall()
        assert node.cmd_pub.publish.call_args.args[0].linear.x == 0.
        node.depth[:] = 1.3
        node.follow_wall()
        assert node.cmd_pub.publish.call_args.args[0].linear.x > 0.


def test_wall_controller_converges_to_parallel_flight(node):
    node.position[:2] = [0., 1.2]
    node.yaw = -0.25
    enter_wall_follow(node)
    node.waypoints = [(100., 0.)]
    node.leg_start = node.position[:2].copy()
    node.next_goal_check = 999.
    for step in range(600):
        now = step * 0.1
        sensor_height = node.position[1] - 0.08 * math.cos(node.yaw)
        node.right_depth = side_wall_image(sensor_height, -node.yaw)
        with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=now):
            node.last_right_depth = now
            node.follow_wall()
        cmd = node.cmd_pub.publish.call_args.args[0]
        c, s = math.cos(node.yaw), math.sin(node.yaw)
        node.position[0] += 0.1 * (c * cmd.linear.x - s * cmd.linear.y)
        node.position[1] += 0.1 * (s * cmd.linear.x + c * cmd.linear.y)
        node.yaw += 0.1 * cmd.angular.z
    assert abs(node.yaw) < 0.03
    assert node.position[1] == pytest.approx(0.58, abs=0.08)


@pytest.mark.parametrize('front, expected', [
    (0.49, 'TURN_LEFT'), (0.50, 'TURN_LEFT'), (0.51, 'GOAL')])
def test_obstacle_threshold_is_half_metre(node, front, expected):
    node.state = 'GOAL'
    node.waypoints = [(3., 0.)]
    node.look = Mock(return_value=(True, front))
    node.go_to_goal()
    assert node.state == expected


def test_wall_exit_returns_to_goal_with_clearance_hysteresis(node):
    node.state = 'LOOK_GOAL'
    node.waypoints = [(3., 0.)]
    node.look = Mock(return_value=(True, 0.7))
    node.look_goal()
    assert node.state == 'GOAL'
    node.go_to_goal()
    assert node.state == 'GOAL'


def test_default_mission_only_targets_origin(node):
    assert node.waypoints == [(0., 0.)]


def test_arrival_holds_exact_target_instead_of_arrival_position(node):
    node.state = 'GOAL'
    node.position[:2] = [0.6 * node.p['tolerance'], 0.]
    node.go_to_goal()
    assert node.state == 'DONE'
    assert np.allclose(node.hold_xy, [0., 0.])
    cmd = node.cmd_pub.publish.call_args.args[0]
    assert cmd.linear.x < 0.


def test_done_corrects_xy_and_altitude_without_depth(node):
    node.state = 'GOAL'
    node.go_to_goal()
    node.position[:] = [0.2, -0.1, 1.4]
    node.depth = None
    with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=1.):
        node.last_odom = 1.
        node.control()
    cmd = node.cmd_pub.publish.call_args.args[0]
    assert node.state == 'DONE'
    assert cmd.linear.x < 0. and cmd.linear.y > 0. and cmd.linear.z > 0.


def test_hold_stops_at_origin_and_does_not_restart_waypoint(node):
    node.state = 'GOAL'
    node.go_to_goal()
    for _ in range(3):
        with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=1.):
            node.last_odom = 1.
            node.control()
        assert node.state == 'DONE'
        cmd = node.cmd_pub.publish.call_args.args[0]
        assert cmd.linear.x == cmd.linear.y == cmd.angular.z == 0.


def test_failed_save_retries_until_success(node, tmp_path):
    from nav_msgs.msg import OccupancyGrid

    node.save_map_path = str(tmp_path / 'arena.npz')
    node.hold_xy = np.array([0., 0.])
    node.done_since = 1.
    node.saved_grid = OccupancyGrid()
    node.saved_grid.header.frame_id = 'odom'
    with patch('drone_vision_control.rgbd_navigation.save_odom_snapshot',
               side_effect=OSError('disk error')):
        node.finish_mapping(3.)
    assert not node.snapshot_saved
    with patch('drone_vision_control.rgbd_navigation.save_odom_snapshot',
               return_value=node.save_map_path):
        node.finish_mapping(6.)
    assert node.snapshot_saved


def test_missing_map_is_not_marked_saved(node, tmp_path):
    node.save_map_path = str(tmp_path / 'arena.npz')
    node.hold_xy = np.array([0., 0.])
    node.done_since = 1.
    node.finish_mapping(5.)
    assert not node.snapshot_saved


def test_stationary_cached_map_is_saved_even_without_republication(node, tmp_path):
    from nav_msgs.msg import OccupancyGrid
    from drone_vision_control.astar_navigation import load_snapshot

    node.save_map_path = str(tmp_path / 'arena.npz')
    node.hold_xy = np.array([0., 0.])
    node.done_since = 1.
    msg = OccupancyGrid()
    msg.header.frame_id = 'odom'
    msg.info.width = msg.info.height = 3
    msg.info.resolution = .1
    msg.info.origin.orientation.w = 1.
    msg.data = [0] * 9
    node.saved_grid = msg
    node.finish_mapping(60.)
    assert node.snapshot_saved
    assert load_snapshot(node.save_map_path).info.width == 3


def test_tilted_transform_no_longer_prevents_saving(node, tmp_path):
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid

    node.save_map_path = str(tmp_path / 'tilted.npz')
    node.hold_xy = np.array([0., 0.])
    node.done_since = 1.
    node.saved_grid = OccupancyGrid()
    node.saved_grid.header.frame_id = 'map'
    node.saved_grid.info.width = node.saved_grid.info.height = 5
    node.saved_grid.info.resolution = .1
    node.saved_grid.info.origin.orientation.w = 1.
    node.saved_grid.data = [0] * 24 + [100]
    tf = TransformStamped()
    tf.transform.rotation.y = math.sin(.1)
    tf.transform.rotation.w = math.cos(.1)
    node.snapshot_tf = Mock()
    node.snapshot_tf.lookup_transform.return_value = tf
    node.finish_mapping(60.)
    assert node.snapshot_saved
    assert (tmp_path / 'tilted.npz').is_file()
    assert node.snapshot_tf.lookup_transform.call_args.args[2].nanoseconds == 0


@pytest.mark.parametrize('right', [False, True])
@pytest.mark.parametrize('encoding, value', [('16UC1', 1500), ('32FC1', 1.5)])
def test_depth_cameras_use_metres_and_independent_freshness(node, right, encoding, value):
    node.bridge = Mock()
    node.bridge.imgmsg_to_cv2.return_value = np.full((8, 8), value)
    initial_count = node.depth_count
    with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=5.):
        node.depth_cb(Mock(encoding=encoding), right=right)
    depth = node.right_depth if right else node.depth
    assert np.allclose(depth, 1.5)
    assert node.depth_count == initial_count + (not right)
    assert node.last_right_depth == (5. if right else 0.)
    assert node.last_depth == (0. if right else 5.)


def test_takeoff_starts_goal_without_waiting_for_map(node):
    with patch('drone_vision_control.rgbd_navigation.time.monotonic', return_value=1.):
        node.last_odom = 1.
        node.control()
    assert node.state == 'GOAL'
    assert node.index == 0
    assert np.allclose(node.leg_start, node.position[:2])


@pytest.mark.parametrize('position, expected', [
    ((2., 0.1), True),      # Back on the segment with progress.
    ((2., 0.4), False),     # Too far from the segment.
    ((-1., 0.), False),     # Behind the leg start.
    ((5., 0.), False),      # Beyond the goal.
    ((0.1, 0.), False),     # Insufficient progress past the obstacle.
])
def test_wall_exit_requires_goal_segment_and_progress(position, expected):
    from drone_vision_control.module_rgbd.geometry import on_goal_line

    assert bool(on_goal_line(position, (0., 0.), (4., 0.), 4.)) is expected


def test_zero_length_goal_segment_does_not_trigger_wall_exit():
    from drone_vision_control.module_rgbd.geometry import on_goal_line

    assert not on_goal_line((0., 0.), (0., 0.), (0., 0.), 1.)
