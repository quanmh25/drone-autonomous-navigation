"""Exercise map validation, route invalidation and control stop conditions."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from drone_vision_control.astar_navigation import AStarNavigation
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest


@pytest.fixture
def node():
    with patch('rclpy.node.Node.__init__', return_value=None), \
            patch.object(AStarNavigation, 'declare_parameter'), \
            patch.object(AStarNavigation, 'get_parameter',
                         side_effect=lambda name: SimpleNamespace(value={
                             'goal_x': 1., 'goal_y': 1., 'altitude': 1.5,
                             'clearance': .35, 'speed': .18, 'goal_tolerance': .1,
                             'map_file': ''}[name])), \
            patch.object(AStarNavigation, 'create_publisher'), \
            patch.object(AStarNavigation, 'create_subscription'), \
            patch.object(AStarNavigation, 'create_timer'), \
            patch.object(AStarNavigation, 'get_logger', return_value=Mock()), \
            patch('drone_vision_control.astar_navigation.Buffer'), \
            patch('drone_vision_control.astar_navigation.TransformListener'):
        nav = AStarNavigation()
        nav.get_clock = Mock(return_value=Mock(now=Mock(return_value=Mock(
            nanoseconds=10**9, to_msg=Mock(return_value=OccupancyGrid().header.stamp)))))
        nav.position = SimpleNamespace(x=0., y=0., z=1.5)
        yield nav


def grid_message():
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = .1
    msg.info.width = msg.info.height = 50
    msg.info.origin.position.x = msg.info.origin.position.y = -2.
    msg.info.origin.orientation.w = 1.
    msg.data = [0] * 2500
    return msg


def test_map_plans_to_requested_world_goal(node):
    node.map_cb(grid_message())
    node.plan(np.array([0., 0.]))
    assert node.route[-1] == (1., 1.)
    assert node.path_pub.publish.call_args.args[0].header.frame_id == 'map'


def test_changed_map_invalidates_route(node):
    msg = grid_message()
    node.map_cb(msg)
    node.plan(np.array([0., 0.]))
    assert node.route
    msg.data[30 * 50 + 30] = 100
    node.map_cb(msg)
    assert not node.route
    node.plan(np.array([0., 0.]))
    assert not node.route


def test_identical_map_does_not_discard_route(node):
    msg = grid_message()
    node.map_cb(msg)
    node.plan(np.array([0., 0.]))
    node.map_cb(msg)
    assert node.route


def test_missing_map_hovers_after_takeoff(node):
    node.state = 'NAVIGATE'
    with patch('drone_vision_control.astar_navigation.time.monotonic', return_value=1.):
        node.last_odom = 1.
        node.control()
    msg = node.cmd_pub.publish.call_args.args[0]
    assert msg.linear.x == msg.linear.y == msg.angular.z == 0.


def test_stale_odom_stops_all_motion(node):
    with patch('drone_vision_control.astar_navigation.time.monotonic', return_value=10.):
        node.control()
    msg = node.cmd_pub.publish.call_args.args[0]
    assert msg.linear.x == msg.linear.y == msg.linear.z == msg.angular.z == 0.


def test_unknown_goal_cannot_create_route(node):
    msg = grid_message()
    msg.data[30 * 50 + 30] = -1
    node.map_cb(msg)
    node.plan(np.array([0., 0.]))
    assert node.route == []
