from setuptools import setup
import os
from glob import glob

package_name = 'wrs25_pose_estimation_module_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'action'), glob('action/*.action')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='ROS 2 action package for WRS Vision box and spray models',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'box_action_server = wrs25_pose_estimation_module_ros2.box_action_server:main',
            'spray_action_server = wrs25_pose_estimation_module_ros2.spray_action_server:main',
            'vision_check_server = wrs25_pose_estimation_module_ros2.vision_check_server:main',
        ],
    },
)

