#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    # Declare arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyTHS1',
        description='Serial port for STM32'
    )
    
    # STM32 Driver Node
    stm32_driver = Node(
        package='car_hardware',
        executable='stm32_driver',
        name='stm32_driver_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': 115200,
            'wheel_base': 0.25,
            'max_speed': 1.0,
            'max_steering_angle': 40.0
        }]
    )
    
    # Teleop Keyboard Node
    teleop_keyboard = Node(
        package='car_hardware',
        executable='teleop_keyboard',
        name='teleop_keyboard_node',
        output='screen',
        parameters=[{
            'speed_step': 0.1,
            'turn_step': 0.3,
            'max_speed': 1.0,
            'max_turn': 1.0
        }],
        prefix='xterm -e'  # Run in separate terminal
    )
    
    return LaunchDescription([
        serial_port_arg,
        stm32_driver,
        # teleop_keyboard,  # Uncomment to auto-launch teleop
    ])
