#!/usr/bin/env python3
"""
ROS 2 Action Server for VisionCheck action (unified interface for behavior tree).
This server wraps the BoxVision and SprayVision models to provide a unified interface.
"""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose

from wrs25_pose_estimation_module_ros2.box_model_wrapper import BoxModelWrapper
from wrs25_pose_estimation_module_ros2.spray_model_wrapper import SprayModelWrapper
from wrs25_pose_estimation_module_ros2.action import VisionCheck


class VisionCheckServer(Node):
    """Unified action server for Vision Check processing."""
    
    def __init__(self):
        super().__init__('vision_check_server')
        
        # Declare parameter for camera topic
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        
        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # Subscribe to camera topic
        self.image_subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        # Initialize vision model wrappers
        self.box_wrapper = BoxModelWrapper()
        self.spray_wrapper = SprayModelWrapper()
        
        # Create action server
        self._action_server = ActionServer(
            self,
            VisionCheck,
            'vision_check',
            self.execute_callback
        )
        
        self.get_logger().info(f'VisionCheck Action Server started, subscribed to {camera_topic}')
    
    def image_callback(self, msg):
        """Callback to store the latest camera frame."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')
    
    def execute_callback(self, goal_handle):
        """Execute the action goal."""
        camera_id = goal_handle.request.camera_id
        query = goal_handle.request.query
        
        self.get_logger().info(f'Received vision check: camera_id="{camera_id}", query="{query}"')
        
        # Check if we have a frame
        if self.latest_frame is None:
            result_msg = VisionCheck.Result()
            result_msg.success = False
            result_msg.message = 'No camera frame available yet'
            result_msg.target_pose = Pose()
            goal_handle.succeed()
            return result_msg
        
        # Publish initial feedback
        feedback_msg = VisionCheck.Feedback()
        feedback_msg.status = 'Processing camera frame...'
        goal_handle.publish_feedback(feedback_msg)
        
        # Route to appropriate handler based on query
        if 'bottle' in query.lower() or 'spray' in query.lower():
            result = self.process_spray_query(query, goal_handle)
        elif 'box' in query.lower() or 'container' in query.lower():
            result = self.process_box_query(query, goal_handle)
        else:
            # Default: try box detection
            result = self.process_box_query(query, goal_handle)
        
        # Create result message
        result_msg = VisionCheck.Result()
        result_msg.success = result['success']
        result_msg.message = result['message']
        result_msg.target_pose = result.get('target_pose', Pose())
        
        goal_handle.succeed()
        return result_msg
    
    def process_spray_query(self, query, goal_handle):
        """Process spray/bottle detection queries."""
        feedback_msg = VisionCheck.Feedback()
        feedback_msg.status = 'Running spray detection...'
        goal_handle.publish_feedback(feedback_msg)
        
        result = self.spray_wrapper.process_frame(self.latest_frame)
        
        if not result['success']:
            return {
                'success': False,
                'message': result['message'],
                'target_pose': Pose()
            }
        
        objects = result.get('objects', [])
        
        if len(objects) == 0:
            return {
                'success': False,
                'message': 'No spray bottles detected',
                'target_pose': Pose()
            }
        
        # Get the first detected object
        obj = objects[0]
        
        # Create pose message
        pose = Pose()
        
        # Use centroid as position (convert pixel coordinates to meters, adjust as needed)
        # TODO: Calibrate pixel-to-meter conversion based on camera setup
        centroid = obj.get('centroid', [0, 0])
        pose.position.x = float(centroid[0]) / 1000.0  # Convert to meters
        pose.position.y = float(centroid[1]) / 1000.0
        pose.position.z = 0.0  # Height would need depth camera data
        
        # Use orientation for quaternion (convert degrees to quaternion)
        orientation_deg = obj.get('orientation_deg', 0.0)
        orientation_rad = np.deg2rad(orientation_deg)
        
        # Simple 2D rotation around Z-axis
        pose.orientation.w = np.cos(orientation_rad / 2.0)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = np.sin(orientation_rad / 2.0)
        
        self.get_logger().info(
            f'Spray bottle detected at pixel coords [{centroid[0]}, {centroid[1]}], '
            f'orientation: {orientation_deg:.1f}°'
        )
        
        return {
            'success': True,
            'message': f'Detected {len(objects)} spray bottle(s)',
            'target_pose': pose
        }
    
    def process_box_query(self, query, goal_handle):
        """Process box/container detection queries."""
        feedback_msg = VisionCheck.Feedback()
        feedback_msg.status = 'Running box detection...'
        goal_handle.publish_feedback(feedback_msg)
        
        result = self.box_wrapper.process_frame(self.latest_frame)
        
        if not result['success']:
            return {
                'success': False,
                'message': result['message'],
                'target_pose': Pose()
            }
        
        objects = result.get('objects', [])
        
        if len(objects) == 0:
            return {
                'success': False,
                'message': 'No boxes detected',
                'target_pose': Pose()
            }
        
        # Get the first detected object
        obj = objects[0]
        
        # Create pose message
        pose = Pose()
        
        # Use center as position (convert pixel coordinates to meters)
        # TODO: Calibrate pixel-to-meter conversion based on camera setup
        center = obj.get('center', (0, 0))
        pose.position.x = float(center[0]) / 1000.0  # Convert to meters
        pose.position.y = float(center[1]) / 1000.0
        pose.position.z = 0.0  # Height would need depth camera data
        
        # Use angle for quaternion (convert degrees to quaternion)
        angle_deg = obj.get('angle', 0.0)
        angle_rad = np.deg2rad(angle_deg)
        
        # Simple 2D rotation around Z-axis
        pose.orientation.w = np.cos(angle_rad / 2.0)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = np.sin(angle_rad / 2.0)
        
        self.get_logger().info(
            f'Box detected at pixel coords [{center[0]}, {center[1]}], '
            f'angle: {angle_deg:.1f}°'
        )
        
        return {
            'success': True,
            'message': f'Detected {len(objects)} box(es)',
            'target_pose': pose
        }


def main(args=None):
    rclpy.init(args=args)
    
    vision_check_server = VisionCheckServer()
    
    try:
        rclpy.spin(vision_check_server)
    except KeyboardInterrupt:
        pass
    finally:
        vision_check_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
