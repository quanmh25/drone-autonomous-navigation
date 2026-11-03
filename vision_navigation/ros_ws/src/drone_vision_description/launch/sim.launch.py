import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare



def generate_launch_description():
    """Publish camera optical TF even when simulation is launched on its own."""
    pkg_drone_arena = get_package_share_directory('drone_vision_description')

    models_path = os.path.join(pkg_drone_arena, 'models')
    workspace_share = os.path.dirname(pkg_drone_arena)
    gz_resource_path = models_path + ':' + workspace_share

    set_env = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=gz_resource_path
    )

    world_path = PathJoinSubstitution([FindPackageShare('drone_vision_description'), 'worlds', 'arena.sdf'])
    default_xacro_file = os.path.join(pkg_drone_arena, 'urdf', 'drone_camera.xacro')
    rviz_config_file = os.path.join(pkg_drone_arena, 'config', 'model.rviz')

    declare_urdf_model_cmd = DeclareLaunchArgument(
        'urdf_model',
        default_value=default_xacro_file,
        description='Full path to the Xacro file'
    )


    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        output='screen'
    )


    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': ParameterValue(
                Command(['xacro ', LaunchConfiguration('urdf_model')]), value_type=str)}
        ]
    )


    spawn_drone = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'parrot_bebop',
            '-x', LaunchConfiguration('spawn_x'),
            '-y', LaunchConfiguration('spawn_y'), '-z', '0.03'
        ],
        output='screen'
    )


    bridge_config_file = os.path.join(pkg_drone_arena, 'config', 'ros_gz_bridge.yaml')
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': bridge_config_file,
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
        }],
        output='screen'
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', LaunchConfiguration('rviz_config'),
                   '-f', LaunchConfiguration('fixed_frame')]
    )


    return LaunchDescription([
        set_env,
        declare_urdf_model_cmd,
        DeclareLaunchArgument('fixed_frame', default_value='odom'),
        DeclareLaunchArgument('spawn_x', default_value='-3.8'),
        DeclareLaunchArgument('spawn_y', default_value='-3.8'),
        DeclareLaunchArgument('rviz_config', default_value=rviz_config_file),
        gz_sim,
        robot_state_publisher,
        spawn_drone,
        bridge,
        rviz2
    ])
