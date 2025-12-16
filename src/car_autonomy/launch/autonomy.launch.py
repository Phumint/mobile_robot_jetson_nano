from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    # 1. Include your EXISTING driver from car_bringup
    # This ensures we use the correct ports and settings you already have
    bringup_dir = get_package_share_directory('car_bringup')
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup_dir, 'launch', 'sensors.launch.py')),
        launch_arguments={'serial_port': '/dev/ttyTHS1'}.items()
    )

    return LaunchDescription([
        # Start the Hardware
        sensors_launch,

        # Start Lane Detection
        Node(
            package='car_autonomy',
            executable='lane_detection',
            name='lane_detector',
            output='screen',
            parameters=[{'record': False, 't_section_turn': 'left'}]
        ),

        # Start Lane Follower (Controller)
        Node(
            package='car_autonomy',
            executable='lane_follower',
            name='lane_follower',
            output='screen',
            parameters=[{'max_speed_ms': 1.0}] # Start slow!
        )
    ])
