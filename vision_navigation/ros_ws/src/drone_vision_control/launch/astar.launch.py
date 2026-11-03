"""Start a fresh simulation and A* using the grid saved by an earlier run."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from drone_vision_control.astar_navigation import default_map_directory, load_snapshot
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node



def generate_launch_description():
    """Validate the saved map, then start simulation and A* navigation."""
    pkg_sim = get_package_share_directory('drone_vision_description')
    pkg_control = get_package_share_directory('drone_vision_control')
    config_directory = default_map_directory(pkg_control)

    snapshot = Path(config_directory) / 'arena_latest.npz'
    spawn_x, spawn_y = -3.0, -0.5
    goal_x, goal_y = 3.5, -1.0


    if not snapshot.is_file():
        raise RuntimeError(
            f'No saved map: {snapshot}. Run run.launch.py and wait for "Map saved" first.')
    grid = load_snapshot(snapshot)
    if grid.header.frame_id != 'odom':
        raise RuntimeError('Saved map must be frozen in odom for this simulation restart')


    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_sim, 'launch', 'sim.launch.py')),
        launch_arguments={
            'fixed_frame': 'odom',
            'spawn_x': str(spawn_x),
            'spawn_y': str(spawn_y),
            'rviz_config': os.path.join(pkg_sim, 'config', 'saved_map.rviz'),
        }.items(),
    )


    planner = Node(
        package='drone_vision_control',
        executable='astar_navigation',
        name='astar_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'map_file': str(snapshot),
            'goal_x': goal_x,
            'goal_y': goal_y,
        }],
    )


    return LaunchDescription([
        LogInfo(msg=f'A* using saved map: {snapshot}'),
        simulation,
        TimerAction(period=7.0, actions=[planner]),
    ])
