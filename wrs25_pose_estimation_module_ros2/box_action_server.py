#!/usr/bin/env python3
"""
ROS 2 Action Server for Box Vision model.
"""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from wrs25_pose_estimation_module_ros2.box_model_wrapper import BoxModelWrapper
from wrs25_pose_estimation_module_ros2.action import BoxVision


class BoxActionServer(Node):
    """Action server for Box Vision processing."""
    
    def __init__(self):
        super().__init__('box_action_server')
        
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
        
        self._action_server = ActionServer(
            self,
            BoxVision,
            'box_vision',
            self.execute_callback
        )
        self.box_wrapper = BoxModelWrapper()
        self.get_logger().info(f'Box Vision Action Server started, subscribed to {camera_topic}')
    
    def image_callback(self, msg):
        """Callback to store the latest camera frame."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')
    
    def execute_callback(self, goal_handle):
        """Execute the action goal."""
        self.get_logger().info('Received goal: processing camera frame')
        
        # Check if we have a frame
        if self.latest_frame is None:
            result_msg = BoxVision.Result()
            result_msg.success = False
            result_msg.message = 'No camera frame available yet'
            result_msg.object_names = []
            result_msg.angles = []
            result_msg.center_x = []
            result_msg.center_y = []
            result_msg.visualization_path = ''
            goal_handle.succeed()
            return result_msg
        
        # Publish feedback
        feedback_msg = BoxVision.Feedback()
        feedback_msg.status = 'Processing camera frame...'
        feedback_msg.progress = 0.1
        goal_handle.publish_feedback(feedback_msg)
        
        # Process image
        feedback_msg.status = 'Processing with box model...'
        feedback_msg.progress = 0.5
        goal_handle.publish_feedback(feedback_msg)
        
        result = self.box_wrapper.process_frame(self.latest_frame)
        
        # Create result message
        result_msg = BoxVision.Result()
        result_msg.success = result['success']
        result_msg.message = result['message']
        
        if result['success']:
            # Extract object data
            objects = result['objects']
            result_msg.object_names = [obj['name'] for obj in objects]
            result_msg.angles = [obj['angle'] for obj in objects]
            result_msg.center_x = [obj['center'][0] for obj in objects]
            result_msg.center_y = [obj['center'][1] for obj in objects]
            result_msg.visualization_path = result['visualization_path'] if result['visualization_path'] else ''
            
            feedback_msg.status = 'Processing complete'
            feedback_msg.progress = 1.0
            goal_handle.publish_feedback(feedback_msg)
            
            self.get_logger().info(f'Successfully processed {len(objects)} objects')
        else:
            self.get_logger().error(f'Processing failed: {result["message"]}')
        
        goal_handle.succeed()
        return result_msg


def main(args=None):
    rclpy.init(args=args)
    
    box_action_server = BoxActionServer()
    
    try:
        rclpy.spin(box_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        box_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

