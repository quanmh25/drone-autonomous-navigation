"""Launch simulation, RTAB-Map and RGB-D navigation."""

import os

from ament_index_python.packages import get_package_share_directory
from drone_vision_control.module_rgbd.map_snapshot import default_map_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node



def generate_launch_description():
    """Use Gazebo odometry for mapping without a second odometry publisher."""
    pkg_sim = get_package_share_directory('drone_vision_description')
    pkg_control = get_package_share_directory('drone_vision_control')
    config_directory = default_map_directory(pkg_control)
    pkg_rtabmap = get_package_share_directory('rtabmap_launch')


    # A new file per launch prevents past observations appearing in a new world.
    number = 1
    fresh_database = os.path.join(config_directory, f'arena_{number:03d}.db')
    while os.path.exists(fresh_database):
        number += 1
        fresh_database = os.path.join(config_directory, f'arena_{number:03d}.db')


    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_sim, 'launch', 'sim.launch.py')),
        launch_arguments={'fixed_frame': 'map'}.items(),
    )


    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_rtabmap, 'launch', 'rtabmap.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'map_topic': '/rtabmap/map',
            'args': ('--Grid/3D false --Grid/Sensor 1 --Grid/CellSize 0.05 '
                     '--Grid/MapFrameProjection true --Grid/NormalsSegmentation false '
                     '--Grid/MaxGroundHeight 0.15 --Grid/MaxObstacleHeight 2.5 '
                     '--Grid/RangeMax 8.0 --Grid/RayTracing true'),
            'frame_id': 'base_link',
            'visual_odometry': 'false',
            'odom_topic': '/odom',
            'rgb_topic': '/front_rgbd/image',
            'depth_topic': '/front_rgbd/depth_image',
            'camera_info_topic': '/front_rgbd/camera_info',
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.05',
            'qos': '2',
            'database_path': fresh_database,
            'rtabmap_viz': 'false',
            'rviz': 'false',
        }.items(),
    )


    controller = Node(
        package='drone_vision_control',
        executable='rgbd_navigation',
        name='rgbd_navigation',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )


    return LaunchDescription([
        LogInfo(msg=f'RTAB-Map database: {fresh_database}'),
        simulation,
        mapping,
        TimerAction(period=7.0, actions=[controller]),
    ])
