#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyTHS1',
        description='Serial port for STM32'
    )
    
    # STM32 Driver Node (sensors only mode)
    stm32_driver = Node(
        package='car_hardware',
        executable='stm32_driver',
        name='stm32_driver_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': 115200
        }]
    )
    
    return LaunchDescription([
        serial_port_arg,
        stm32_driver,
    ])
