# Drone Autonomous Navigation

Two ROS 2 projects for autonomous drone navigation, obstacle avoidance, mapping, and navigation to a specified destination in Gazebo.

| Project | Navigation sensors | Mapping system | Reactive obstacle avoidance |
|---|---|---|---|
| `lidar_slam_navigation` | LiDAR | SLAM Toolbox | Left-wall following |
| `vision_navigation` | Front and right RGB-D cameras | RTAB-Map | Right-wall following |

Both projects use Gazebo odometry to estimate the drone's position and heading. Velocity commands are published on `/cmd_vel` and applied by the Gazebo VelocityControl plugin. These are simulation controllers, not flight controllers for a physical drone.

## 1. The two operating modes

### `run.launch.py`: autonomous flight and mapping

This launch starts Gazebo, RViz, the mapping system, and the reactive waypoint controller.

The drone:

1. Waits for odometry and takes off to the configured altitude, normally **1.5 m**.
2. Flies toward the waypoints defined in the navigation node.
3. Uses its sensors to avoid obstacles and follow walls when the direct route is blocked.
4. Supplies sensor observations to SLAM while moving, building a map of the observed environment.
5. Holds position at the final waypoint and automatically saves the map after settling.
6. Continues hovering after saving. It does not automatically land or terminate the node.

**`run` combines autonomous waypoint navigation with mapping. It does not autonomously explore every part of the environment.** Map coverage depends on what the sensors observe during the flight. Add suitable waypoints if more coverage is needed.

### `astar.launch.py`: autonomous navigation using the saved map

This launch starts a new simulation and loads the map produced by an earlier mapping run. It does not start SLAM again.

The drone:

1. Loads the saved `arena_latest.npz` map. The launch checks the map before starting Gazebo.
2. Spawns at the configured starting position and takes off.
3. Inflates obstacles to provide clearance around the vehicle. Unknown cells are also blocked.
4. Uses A* to find a route through known free space to the requested destination.
5. Follows that route while checking live sensor data for unexpected obstacles or missing observations.
6. Holds position and altitude at the destination without automatically landing.

The start and destination must be inside traversable map regions and connected by a free route. A location that looks empty in Gazebo may still be unknown or blocked in the saved map.

### Recommended workflow

```text
run.launch.py
    → autonomous waypoint flight + SLAM mapping
    → final waypoint reached + "Map saved" message
    → stop the old simulation
astar.launch.py
    → load the saved map
    → plan a route to the configured destination
    → autonomous route following + position hold
```

Run these modes separately: each launch starts its own simulation and controller.

## 2. Repository structure

```text
drone-autonomous-navigation/
├── README.md
├── lidar_slam_navigation/
│   └── ros_ws/src/
│       ├── drone_lidar_description/    # Drone model, LiDAR, world, bridge, simulation
│       └── drone_lidar_control/
│           ├── launch/                # run.launch.py and astar.launch.py
│           ├── config/                # SLAM Toolbox and RViz configuration
│           ├── map/                   # Saved LiDAR maps
│           └── drone_lidar_control/
│               ├── lidar_navigation.py
│               ├── astar_navigation.py
│               └── module_lidar/      # Scan processing and map export
└── vision_navigation/
    └── ros_ws/src/
        ├── drone_vision_description/   # Drone model, cameras, world, bridge, RViz
        └── drone_vision_control/
            ├── launch/                # run.launch.py and astar.launch.py
            ├── config/                # Saved maps and RTAB-Map databases
            └── drone_vision_control/
                ├── rgbd_navigation.py
                ├── astar_navigation.py
                └── module_rgbd/       # Depth processing, geometry, map export
```

Within each project, the reactive controller and the A* controller do not import or inherit from one another. Their connection is the saved map file.

## 3. Requirements and building

The commands below target **ROS 2 Jazzy** on Linux with Gazebo available through `gz sim`. A graphical environment is needed to view Gazebo and RViz.

Main dependencies include:

- `ros_gz_sim` and `ros_gz_bridge` for simulation integration.
- `robot_state_publisher`, `xacro`, `tf2_ros`, and `rviz2`.
- NumPy and PyYAML.
- SLAM Toolbox for the LiDAR project.
- RTAB-Map and `cv_bridge` for the RGB-D project.

ROS package dependencies are declared in the package manifests. The following `rosdep` commands assume that rosdep is already installed and initialized.

Each build sequence starts from the repository root.

### Build the LiDAR workspace

```bash
cd lidar_slam_navigation/ros_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install
source install/setup.bash
```

### Build the RGB-D workspace

```bash
cd vision_navigation/ros_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install
source install/setup.bash
```

Use separate terminals for the two workspaces and source the appropriate workspace before launching. Run only one simulation at a time: both projects use shared topic names such as `/odom`, `/cmd_vel`, and `/clock`.

## 4. LiDAR project

Run the following commands from `lidar_slam_navigation/ros_ws`.

### Step 1: fly autonomously and build the map

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch drone_lidar_control run.launch.py
```

This starts the simulation, SLAM Toolbox, RViz, and `lidar_navigation`.

The controller flies toward its configured waypoints. When an obstacle blocks the direct route, it follows the obstacle with the wall on its left and periodically checks whether it can resume moving toward the goal.

Raw LiDAR observations arrive on `/scan`. After takeoff, scans are forwarded to `/slam/scan` when the drone is near its working altitude, allowing SLAM to build a horizontal map. The controller observes `/map` for saving; the map does not drive its reactive obstacle avoidance.

At the final waypoint, wait for:

```text
Map saved: .../arena_latest.npz
```

The default source-workspace output directory is:

```text
lidar_slam_navigation/ros_ws/src/drone_lidar_control/map/
├── arena_latest.npz
├── arena_latest.pgm
└── arena_latest.yaml
```

Press `Ctrl+C` and wait for the old simulation to stop before starting A*.

### Step 2: navigate to a destination using the map

```bash
ros2 launch drone_lidar_control astar.launch.py
```

The A* node loads the saved LiDAR map and plans in the `odom` frame. Live LiDAR measurements remain available as an obstacle interlock, but SLAM Toolbox is not launched in this mode.

After A* produces a grid route, a line-of-sight shortcut step connects visible route points through the inflated free grid. This removes unnecessary grid bends in open areas. Every shortcut checks the cells it touches, including cells adjacent to diagonal corners.

This improves the grid route but does not guarantee a globally shortest path in continuous space.

### Change the mapping waypoints or A* destination

The paths below are relative to `lidar_slam_navigation/ros_ws/src/`.

| Setting | File | Configuration |
|---|---|---|
| Mapping-flight waypoints | `drone_lidar_control/drone_lidar_control/lidar_navigation.py` | `self.waypoints` |
| Mapping altitude, speed, wall clearance | Same file | Direct attributes in `__init__` |
| A* destination | `drone_lidar_control/drone_lidar_control/astar_navigation.py` | `self.goal` |
| A* obstacle inflation | Same file | `self.clearance` |
| A* spawn position | `drone_lidar_control/launch/astar.launch.py` | `spawn_x`, `spawn_y` |
| Default mapping spawn position | `drone_lidar_description/launch/sim.launch.py` | Default `spawn_x`, `spawn_y` values |

For example, edit the destination inside the A* node:

```python
self.goal = np.array([1.0, 3.5])
```

This is an absolute XY destination in `odom`, measured in metres. It is not an offset added to the spawn position.

The LiDAR controllers use directly assigned flight settings. Their launch files pass `use_sim_time` to the nodes rather than waypoint or destination parameters.

## 5. RGB-D project

Run the following commands from `vision_navigation/ros_ws`.

### Step 1: fly autonomously and build the map

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch drone_vision_control run.launch.py
```

This starts the simulation, RTAB-Map, RViz, and `rgbd_navigation`.

The two cameras have different roles:

| Camera | Role |
|---|---|
| Front RGB-D | RGB, depth, and calibration feed RTAB-Map; depth also checks the direction of travel |
| Right RGB-D | Depth estimates the right wall's distance and direction; RGB is available for viewing in RViz |

The right camera is not an input to SLAM. RTAB-Map uses Gazebo `/odom`; the launch disables a separate visual odometry process.

The reactive controller moves toward its waypoints and follows walls on the right when necessary. It does not use the occupancy map to plan an A* route during this mapping flight.

At the final waypoint, the drone holds position, waits to settle, and exports the map. Wait for `Map saved` before stopping the simulation.

Default output files are located in:

```text
vision_navigation/ros_ws/src/drone_vision_control/config/
├── arena_latest.npz
├── arena_latest.pgm
├── arena_latest.yaml
└── arena_001.db, arena_002.db, ...
```

Each mapping launch selects an unused numbered RTAB-Map database to keep observations from separate runs apart. The `arena_latest.*` files are replaced when a new mapping flight successfully saves its snapshot.

### Step 2: navigate to a destination using the map

Stop the mapping launch with `Ctrl+C`, then run:

```bash
ros2 launch drone_vision_control astar.launch.py
```

The launch loads `arena_latest.npz`, verifies that it is frozen in `odom`, starts a new simulation, and runs A*. RTAB-Map is not restarted. Front-camera depth remains active to detect unexpected obstacles.

The RGB-D A* implementation currently preserves grid-path bends. The line-of-sight shortcut step in the LiDAR A* implementation has not been added to this version.

### Change the mapping waypoints or A* destination

The paths below are relative to `vision_navigation/ros_ws/src/`.

| Setting | File | Configuration |
|---|---|---|
| Mapping-flight waypoints | `drone_vision_control/drone_vision_control/rgbd_navigation.py` | `self.waypoints` |
| Mapping altitude, speed, safe distance | Same file | `self.p` dictionary |
| A* destination | `drone_vision_control/launch/astar.launch.py` | `goal_x`, `goal_y` |
| A* spawn position | Same launch file | `spawn_x`, `spawn_y` |
| Saved map used by A* | Same launch file | `snapshot` |
| Default mapping spawn position | `drone_vision_description/launch/sim.launch.py` | Default `spawn_x`, `spawn_y` values |

For example, inside the RGB-D A* launch:

```python
spawn_x, spawn_y = -3.0, -0.5
goal_x, goal_y = 3.5, -1.0
```

The saved-map launch supplies the destination and map file to the RGB-D A* node. With this launch, the destination is interpreted in the saved map's `odom` frame.

## 6. Map formats, coordinates, and limitations

- **NPZ:** lossless occupancy values and map geometry used by the A* controller.
- **PGM and YAML:** an image export and its metadata for inspection or use with other map tools.
- **RTAB-Map DB:** SLAM database; the current A* launch reads the NPZ snapshot, not the database.
- The two projects use different NPZ schemas. Do not directly substitute a LiDAR map for an RGB-D map or vice versa.
- Map-directory helpers prefer source-workspace storage and fall back to installed package storage. Check the actual output path in the `Map saved` message.
- Maps are frozen in `odom`. A simulation restart must preserve the environment and coordinate convention. The system relies on simulation odometry; it does not implement real-world relocalization against a saved map.
- Rebuild the map if the world or obstacle arrangement changes.
- Planning is two-dimensional in XY at the configured flight altitude. A* does not search for routes above or below obstacles.
- Unknown cells and cells within the inflated obstacle margin are not traversable.
- A live obstacle stop does not imply that the static map is updated or that a new detour will automatically become available.
- Successful mapping-flight completion does not guarantee that the map covers every later A* destination.

## 7. Visualization and troubleshooting

### Useful RViz topics

| Topic | Content |
|---|---|
| `/odom` | Drone odometry |
| `/cmd_vel` | Velocity commands |
| `/scan` | Raw LiDAR scan |
| `/slam/scan` | LiDAR scan forwarded for mapping |
| `/map` | Live LiDAR map or loaded LiDAR A* map |
| `/rtabmap/map` | Live RGB-D map |
| `/astar/saved_map` | Loaded RGB-D A* map |
| `/astar/path` | Planned A* route |
| `/front_rgbd/image`, `/right_rgbd/image` | Front and right colour images |
| `/front_rgbd/depth_image`, `/right_rgbd/depth_image` | Depth images, not colour images |

### A* cannot find the saved map

Run the matching project's `run.launch.py`, allow the drone to reach its final waypoint, and wait for `Map saved`. Seeing a map in RViz does not necessarily mean the NPZ file has been written yet.

### A* cannot find a route

Check that the start and destination are inside known free space. Also check obstacle inflation: a passage that appears open in the displayed occupancy map may become blocked after adding vehicle clearance.

If the required area is unknown, adjust the mapping waypoints and perform another mapping flight.

### TF reports backward time jumps or RViz repeatedly resets

Stop old launches and inspect remaining simulation processes and clock publishers:

```bash
pgrep -af 'gz sim|gz-sim|parameter_bridge|ros2 launch'
ros2 topic info /clock --verbose
```

One Gazebo session can include a parent process, server, and GUI. Multiple process lines alone do not prove that multiple clocks exist. Avoid running multiple worlds or controllers sharing the same ROS topics.

### Edited settings do not appear when launching

Build and source the correct workspace again:

```bash
colcon build --symlink-install
source install/setup.bash
```

Restart the launch after editing. Do not launch `run` and `astar` together; each already starts its own simulation and controller.
