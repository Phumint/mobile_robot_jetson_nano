from setuptools import setup
import os
from glob import glob

package_name = 'car_autonomy'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- ADD THIS BLOCK HERE ---
        # This tells ROS: "Take everything in the 'launch' folder 
        # and put it where ros2 launch can find it."
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # ---------------------------
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Autonomy package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detection = car_autonomy.lane_detection:main',
            'lane_follower = car_autonomy.lane_follower:main',
        ],
    },
)
