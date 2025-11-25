from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    spray_action_server_node = Node(
        package='wrs25_pose_estimation_module_ros2',
        executable='spray_action_server.py',
        name='spray_action_server_node',
        output='screen',
    )

    # TODO: add box action server node
    # box_action_server_node = Node(
    #     package='wrs25_pose_estimation_module_ros2',
    #     executable='box_action_server.py',
    #     name='box_action_server_node',
    #     output='screen',
    # )

    return LaunchDescription([
        spray_action_server_node,
        # box_action_server_node,
    ])