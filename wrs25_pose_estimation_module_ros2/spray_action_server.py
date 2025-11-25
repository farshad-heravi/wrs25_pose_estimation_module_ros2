#!/usr/bin/env python3
"""
ROS 2 Action Server for Spray Vision model.
"""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from wrs25_pose_estimation_module_ros2.spray_model_wrapper import SprayModelWrapper
from wrs25_pose_estimation_module_ros2.action import SprayVision


class SprayActionServer(Node):
    """Action server for Spray Vision processing."""
    
    def __init__(self):
        super().__init__('spray_action_server')
        
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
            SprayVision,
            'spray_vision',
            self.execute_callback
        )
        self.spray_wrapper = SprayModelWrapper()
        self.get_logger().info(f'Spray Vision Action Server started, subscribed to {camera_topic}')
    
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
            result_msg = SprayVision.Result()
            result_msg.success = False
            result_msg.message = 'No camera frame available yet'
            result_msg.object_ids = []
            result_msg.tail_x = []
            result_msg.tail_y = []
            result_msg.head_x = []
            result_msg.head_y = []
            result_msg.centroid_x = []
            result_msg.centroid_y = []
            result_msg.orientations = []
            result_msg.visualization_path = ''
            goal_handle.succeed()
            return result_msg
        
        # Publish feedback
        feedback_msg = SprayVision.Feedback()
        feedback_msg.status = 'Processing camera frame...'
        feedback_msg.progress = 0.1
        goal_handle.publish_feedback(feedback_msg)
        
        # Process image
        feedback_msg.status = 'Processing with spray model...'
        feedback_msg.progress = 0.3
        goal_handle.publish_feedback(feedback_msg)
        
        feedback_msg.status = 'Running YOLO prediction...'
        feedback_msg.progress = 0.5
        goal_handle.publish_feedback(feedback_msg)
        
        feedback_msg.status = 'Generating masks...'
        feedback_msg.progress = 0.7
        goal_handle.publish_feedback(feedback_msg)
        
        feedback_msg.status = 'Analyzing regions...'
        feedback_msg.progress = 0.9
        goal_handle.publish_feedback(feedback_msg)
        
        result = self.spray_wrapper.process_frame(self.latest_frame)

        # illustrate the result
        # self.spray_wrapper.illustrate_result(self.latest_frame, result['objects'])

        #  postprocess the result
        result = self.postprocess_result(result)

        # select the bottle for picking 
        bottle_for_picking = self.select_bottle_for_picking(result)
        print(f'Bottle for picking: {bottle_for_picking}')
        
        # illustrate the bottle for picking
        self.spray_wrapper.illustrate_result(self.latest_frame, result['objects'], select_bottle=bottle_for_picking)


        # Create result message
        result_msg = SprayVision.Result()
        result_msg.success = result['success']
        result_msg.message = result['message']
        
        if result['success']:
            orientation_rad = np.deg2rad(bottle_for_picking['orientation_deg'])
            
            # Extract object data
            objects = result['objects']
            result_msg.object_ids = [bottle_for_picking['object_id']]
            result_msg.tail_x = [bottle_for_picking['tail'][0]]
            result_msg.tail_y = [bottle_for_picking['tail'][1]]
            result_msg.head_x = [bottle_for_picking['head'][0]]
            result_msg.head_y = [bottle_for_picking['head'][1]]
            result_msg.centroid_x = [bottle_for_picking['centroid'][0]]
            result_msg.centroid_y = [bottle_for_picking['centroid'][1]]
            result_msg.orientations = [orientation_rad]
            result_msg.visualization_path = result['visualization_path'] if result['visualization_path'] else ''
            
            feedback_msg.status = 'Processing complete'
            feedback_msg.progress = 1.0
            goal_handle.publish_feedback(feedback_msg)
            
            self.get_logger().info(f'Successfully processed {len(objects)} objects')
        else:
            self.get_logger().error(f'Processing failed: {result["message"]}')
        
        goal_handle.succeed()
        return result_msg

    def postprocess_result(self, result):
        """Postprocess the result."""
        filtered_results = []

        # check the length of the bottles detected in pixels to filter out the false detections
        bottles_length = [np.abs(np.sqrt((obj['tail'][0] - obj['head'][0])**2 + (obj['tail'][1] - obj['head'][1])**2)) for obj in result['objects']]

        # print(f'Bottles length: {bottles_length}')
        for index, length in enumerate(bottles_length):
            if 60 <= length <= 100:
                filtered_results.append(result['objects'][index].copy())
        result['objects'] = filtered_results
        return result

    def select_bottle_for_picking(self, result):
        """Select the bottle for picking."""
        # select the closest bottle to bottom-left corner of the image
        bottom_left_corner = (0, self.latest_frame.shape[0])
        closest_bottle = min(result['objects'], key=lambda x: np.abs(np.sqrt((x['centroid'][0] - bottom_left_corner[0])**2 + (x['centroid'][1] - bottom_left_corner[1])**2)))
        return closest_bottle

def main(args=None):
    rclpy.init(args=args)
    
    spray_action_server = SprayActionServer()
    
    try:
        rclpy.spin(spray_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        spray_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

