"""Launch LiDAR simulation and A* using the saved odom map."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from drone_lidar_control.astar_navigation import load_map, map_directory


def generate_launch_description():
    description = get_package_share_directory('drone_lidar_description')
    control = get_package_share_directory('drone_lidar_control')

    # Check the same default map used by the A* node before starting Gazebo.
    filename = os.path.join(map_directory(), 'arena_latest.npz')
    try:
        load_map(filename)
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(
            f'Cannot load {filename}; run mapping until "Map saved": {error}') from error

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description, 'launch', 'sim.launch.py')),
        launch_arguments={
            'spawn_x': '-3.7',
            'spawn_y': '-2.5',
            'rviz_config': os.path.join(control, 'config', 'astar.rviz'),
        }.items(),
    )

    planner = Node(
        package='drone_lidar_control',
        executable='astar_navigation',
        name='astar_navigation',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        simulation,
        TimerAction(period=5.0, actions=[planner]),
    ])
