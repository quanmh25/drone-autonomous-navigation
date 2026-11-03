"""Check saved map coordinates, optimal planning and flight interlocks."""

import math
from unittest.mock import Mock, patch

from drone_lidar_control.astar_navigation import (
    astar, AStarNavigation, blocked_grid, grid_cell, load_map)
from drone_lidar_control.module_lidar.map_snapshot import save_map
from geometry_msgs.msg import Transform
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest


def grid():
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = .1
    msg.info.width = msg.info.height = 100
    msg.info.origin.position.x = msg.info.origin.position.y = -5.
    msg.info.origin.orientation.w = 1.
    msg.data = [0]*10000
    return msg


def test_saved_map_composes_slam_offset_and_rotation(tmp_path):
    msg = grid()
    msg.data[0] = -1
    msg.data[1] = 100
    tf = Transform()
    tf.translation.x = 2.
    tf.rotation.z, tf.rotation.w = math.sin(math.pi/4), math.cos(math.pi/4)
    filename = save_map(msg, tmp_path/'arena.npz', tf)
    restored = load_map(filename)
    assert restored.header.frame_id == 'odom'
    assert restored.data == msg.data
    assert restored.info.origin.position.x == pytest.approx(7.)
    assert restored.info.origin.position.y == pytest.approx(-5.)
    assert restored.info.origin.orientation.z == pytest.approx(math.sin(math.pi/4))
    assert (tmp_path/'arena.yaml').exists()
    assert (tmp_path/'arena.pgm').read_bytes().startswith(b'P5\n100 100\n255\n')


def test_unknown_map_is_not_saved(tmp_path):
    msg = grid()
    msg.data = [-1]*10000
    with pytest.raises(ValueError):
        save_map(msg, tmp_path/'arena.npz', Transform())
    assert not (tmp_path/'arena.npz').exists()


def test_rotated_grid_coordinates():
    assert grid_cell((2., 4.), np.array([2., 3., math.pi/2]), .1) == (10, 0)


def test_shortest_path_and_no_corner_cutting():
    cells = astar(np.zeros((10, 10), dtype=bool), (0, 0), (6, 4))
    assert sum(math.dist(a, b) for a, b in zip(cells, cells[1:])) == pytest.approx(
        4*math.sqrt(2)+2)
    assert astar(np.array([[False, True], [True, False]]), (0, 0), (1, 1)) == []


def test_unknown_cells_and_clearance_block_paths():
    data = np.zeros((30, 30))
    data[:, 15] = -1
    blocked = blocked_grid(data, .1, .35)
    assert blocked[15, 13]
    assert astar(blocked, (5, 15), (25, 15)) == []
    assert astar(blocked, (5, 15), (-1, 0)) == []


@pytest.fixture
def node(tmp_path):
    tf = Transform()
    tf.rotation.w = 1.
    save_map(grid(), tmp_path/'arena_latest.npz', tf)
    stamp = OccupancyGrid().header.stamp
    with patch('rclpy.node.Node.__init__', return_value=None), \
            patch('drone_lidar_control.astar_navigation.map_directory',
                  return_value=str(tmp_path)), \
            patch.object(AStarNavigation, 'create_publisher', side_effect=lambda *a: Mock()), \
            patch.object(AStarNavigation, 'create_subscription'), \
            patch.object(AStarNavigation, 'create_timer'), \
            patch.object(AStarNavigation, 'get_clock', return_value=Mock(
                now=Mock(return_value=Mock(to_msg=Mock(return_value=stamp))))), \
            patch.object(AStarNavigation, 'get_logger', return_value=Mock()):
        nav = AStarNavigation()
        nav.position = np.array([-3., -2., 1.5])
        nav.yaw = 0.
        nav.last_odom = nav.last_scan = 10.
        nav.state = 'NAVIGATE'
        yield nav


def test_plan_reaches_exact_requested_goal(node):
    node.plan()
    assert np.allclose(node.route[-1], node.goal)
    assert node.path_pub.publish.call_args.args[0].header.frame_id == 'odom'


def test_stale_odom_stops_all_axes(node):
    with patch('drone_lidar_control.astar_navigation.time.monotonic', return_value=11.):
        node.control()
    msg = node.pub_cmd.publish.call_args.args[0]
    assert msg.linear.x == msg.linear.y == msg.linear.z == msg.angular.z == 0.


@pytest.mark.parametrize('front', [None, .2, 5.])
def test_obstacle_unknown_or_stale_scan_stops_motion(node, front):
    node.route = [np.array([0., -2.])]
    node.look_since = 9.
    node.sector = Mock(return_value=front)
    if front == 5.:
        node.last_scan = 8.
    with patch('drone_lidar_control.astar_navigation.time.monotonic', return_value=10.):
        node.control()
    assert node.pub_cmd.publish.call_args.args[0].linear.x == 0.


def test_final_goal_holds_without_indexing_past_route(node):
    node.position[:2] = node.goal
    with patch('drone_lidar_control.astar_navigation.time.monotonic', return_value=10.):
        node.control()
        node.control()
    assert node.state == 'HOLD'


def test_map_directory_prefers_workspace_source(tmp_path):
    from drone_lidar_control.astar_navigation import map_directory

    package = tmp_path/'src'/'drone_lidar_control'
    (package/'map').mkdir(parents=True)
    (package/'package.xml').write_text('<package/>')
    share = tmp_path/'install'/'drone_lidar_control'/'share'/'drone_lidar_control'
    share.mkdir(parents=True)
    module = 'drone_lidar_control.astar_navigation'
    with patch(module+'.__file__', str(tmp_path/'install'/'module.py')), \
            patch(module+'.get_package_share_directory', return_value=str(share)):
        assert map_directory() == str(package/'map')


def test_astar_does_not_initialize_wall_following_or_slam(node):
    assert not hasattr(node, 'wall_started')
    assert not hasattr(node, 'waypoints')
    assert not hasattr(node, 'slam_scan_pub')
    assert not hasattr(node, 'map_tf')
    topics = [call.args[1] for call in node.create_subscription.call_args_list]
    assert topics == ['/odom', '/scan']


def test_forward_motion_requires_scan_received_after_alignment(node):
    node.route = [node.position[:2] + [1., 0.]]
    node.sector = Mock(return_value=5.)
    with patch('drone_lidar_control.astar_navigation.time.monotonic', return_value=10.):
        node.control()
    with patch('drone_lidar_control.astar_navigation.time.monotonic', return_value=10.3):
        node.control()
        assert node.pub_cmd.publish.call_args.args[0].linear.x == 0.
        node.last_scan = 10.3
        node.control()
    assert node.pub_cmd.publish.call_args.args[0].linear.x > 0.


def test_turning_restarts_scan_wait(node):
    node.route = [node.position[:2] + [0., 1.]]
    node.look_since = 9.
    with patch('drone_lidar_control.astar_navigation.time.monotonic', return_value=10.):
        node.control()
    msg = node.pub_cmd.publish.call_args.args[0]
    assert node.look_since is None
    assert msg.linear.x == 0. and msg.angular.z > 0.


def test_open_space_route_becomes_one_straight_segment():
    from drone_lidar_control.astar_navigation import shorten_path

    blocked = np.zeros((15, 15), dtype=bool)
    cells = astar(blocked, (1, 1), (12, 7))
    assert shorten_path(blocked, cells) == [(1, 1), (12, 7)]


@pytest.mark.parametrize('start, end', [((0, 0), (1, 1)), ((1, 1), (0, 0))])
def test_shortcuts_cannot_touch_blocked_diagonal_corners(start, end):
    from drone_lidar_control.astar_navigation import line_clear

    blocked = np.array([[False, True], [False, False]])
    assert not line_clear(blocked, start, end)


def test_shortcuts_preserve_obstacle_detour_and_reduce_length():
    from drone_lidar_control.astar_navigation import line_clear, shorten_path

    blocked = np.zeros((20, 20), dtype=bool)
    blocked[3:16, 9:11] = True
    cells = astar(blocked, (2, 7), (17, 11))
    shortened = shorten_path(blocked, cells)
    assert shortened[0] == cells[0] and shortened[-1] == cells[-1]
    assert len(shortened) > 2
    assert all(line_clear(blocked, a, b) for a, b in zip(shortened, shortened[1:]))
    length = lambda path: sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    assert length(shortened) <= length(cells)


@pytest.mark.parametrize('start, end', [
    ((1, 1), (1, 8)), ((8, 1), (1, 1)), ((1, 8), (8, 1)), ((3, 3), (3, 3)),
])
def test_line_clear_handles_axes_and_identical_points(start, end):
    from drone_lidar_control.astar_navigation import line_clear

    assert line_clear(np.zeros((10, 10), dtype=bool), start, end)
