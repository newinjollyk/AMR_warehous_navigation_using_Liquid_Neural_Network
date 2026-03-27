import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


WORLD_PATH = "/home/newin/Projects/warehouse/warehouse_aruco_02.sdf"


def generate_launch_description():
    # tugbot_recorder share dir (for your own launch/config files)
    pkg_share = get_package_share_directory("tugbot_recorder")

    # Gazebo Classic launch file (must exist in tugbot_recorder/launch/gazebo.launch.py)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": WORLD_PATH
        }.items(),
    )
    # ROS-Gazebo bridge node
    gz_bridge = Node(
    package="ros_gz_bridge",
    executable="parameter_bridge",
    output="screen",
    arguments=[
        "/world/world_demo/model/tugbot/link/camera_front/sensor/color/image@sensor_msgs/msg/Image@gz.msgs.Image",
        "/world/world_demo/model/tugbot/link/scan_front/sensor/scan_front/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
        "/model/tugbot/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
        # "/cmd_vel_safe@geometry_msgs/msg/Twist@gz.msgs.Twist",
    ],
)


    # Prediction node: publish original cmd_vel but remap it to /lnn_cmd_vel for the safety node
    common_prefix = (
    "xterm -e bash -lc "
    "'source /opt/ros/humble/setup.bash; "
    "source ~/Projects/warehouse/ws_warehouse/install/setup.bash; "
    "exec ros2 run tugbot_recorder "
    )

    lnn = Node(
    package="tugbot_recorder",
    executable="lnn_prediction_01",
    name="lnn_prediction_01",
    output="screen",
    emulate_tty=True,
    prefix=common_prefix + "lnn_prediction_01'"
    )

    safety = Node(
    package="tugbot_recorder",
    executable="safety_node",
    name="safety_node",
    output="screen",
    emulate_tty=True,
    prefix=common_prefix + "safety_node'"
    )


    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        prefix="xterm -e",
        arguments=["-d", "/home/newin/Projects/warehouse/LNN.rviz"],
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        gazebo,
        gz_bridge,
        lnn,
        safety,
        rviz,
    ])
