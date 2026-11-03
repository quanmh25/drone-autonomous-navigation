from setuptools import find_packages, setup

package_name = 'drone_lidar_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/map', ['map/README.md']),
        ('share/' + package_name + '/launch',
         ['launch/run.launch.py', 'launch/astar.launch.py']),
        ('share/' + package_name + '/config',
         ['config/slam.yaml', 'config/slam.rviz', 'config/astar.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='quanmh25',
    maintainer_email='maihoangquan250205@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'astar_navigation = drone_lidar_control.astar_navigation:main',
            'lidar_navigation = drone_lidar_control.lidar_navigation:main',
        ],
    },
)
