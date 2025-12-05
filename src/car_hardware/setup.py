from setuptools import setup
import os
from glob import glob

package_name = 'car_hardware'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Hardware interface for RC car',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stm32_driver = car_hardware.stm32_driver_node:main',
            'teleop_keyboard = car_hardware.teleop_keyboard_node:main',
        ],
    },
)
