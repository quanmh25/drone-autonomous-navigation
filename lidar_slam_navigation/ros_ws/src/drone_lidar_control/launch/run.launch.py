"""Launch LiDAR simulation, SLAM and waypoint navigation."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    description = get_package_share_directory('drone_lidar_description')
    control = get_package_share_directory('drone_lidar_control')
    slam = get_package_share_directory('slam_toolbox')

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description, 'launch', 'sim.launch.py')),
        launch_arguments={
            'rviz_config': os.path.join(control, 'config', 'slam.rviz'),
        }.items(),
    )

    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam, 'launch', 'online_async_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_params_file': os.path.join(control, 'config', 'slam.yaml'),
        }.items(),
    )

    controller = Node(
        package='drone_lidar_control',
        executable='lidar_navigation',
        name='lidar_navigation',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        simulation,
        mapping,
        TimerAction(period=5.0, actions=[controller]),
    ])
