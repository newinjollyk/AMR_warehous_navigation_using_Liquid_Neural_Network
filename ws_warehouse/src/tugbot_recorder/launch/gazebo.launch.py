from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    world = LaunchConfiguration("world")

    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value="",
            description="Absolute path to SDF world file",
        ),

        # Equivalent to: ign gazebo <world>
        ExecuteProcess(
            cmd=["ign", "gazebo", world],
            output="screen",
        ),
    ])
