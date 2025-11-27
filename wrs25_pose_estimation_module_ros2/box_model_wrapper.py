"""
Wrapper module for the Box Vision model.
Interfaces with the box model from WRS-Vision folder.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from rclpy.logging import get_logger

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent.parent
WRS_VISION_ROOT = str(SRC_DIR / 'wrs25_pose_estimation_module')
BOX_MODEL_PATH = str(SRC_DIR / 'wrs25_pose_estimation_module' / 'box')

get_logger().debug(f"WRS_VISION_ROOT: {WRS_VISION_ROOT}")
get_logger().debug(f"BOX_MODEL_PATH: {BOX_MODEL_PATH}")

if not os.path.exists(BOX_MODEL_PATH):
    raise RuntimeError(
        f"Box model not found at: {BOX_MODEL_PATH}\n"
        f"Please set WRS_VISION_ROOT environment variable to point to the WRS-Vision directory.\n"
        f"Example: export WRS_VISION_ROOT=/home/mpt/projects/wrs_vision/WRS-Vision"
    )

sys.path.insert(0, BOX_MODEL_PATH)

# Set the PROJECT_ROOT in config before importing
import config as box_config
box_config.PROJECT_ROOT = BOX_MODEL_PATH
box_config.IMG_DIR = os.path.join(BOX_MODEL_PATH, "test_images")
box_config.OUT_DIR = os.path.join(BOX_MODEL_PATH, "results")
box_config.YOLO_MODEL_PATH = os.path.join(BOX_MODEL_PATH, "weights", "best.pt")

from image_io import yolo_predict_polygons
from region_analysis import RegionAnalyzer


class BoxModelWrapper:
    """Wrapper class for the Box Vision model."""
    
    def __init__(self):
        """Initialize the box model wrapper."""
        analysis_dir = os.path.join(box_config.OUT_DIR, "analysis")
        self.analyzer = RegionAnalyzer(save_dir=analysis_dir)
        print("Box model wrapper initialized")
    
    def process_frame(self, img, image_name="camera_frame"):
        """
        Process an image frame directly from memory.
        
        Args:
            img (numpy.ndarray): Input image as numpy array (OpenCV format)
            image_name (str): Name to use for saving results
            
        Returns:
            dict: Dictionary containing:
                - success (bool): Whether processing was successful
                - message (str): Status message
                - objects (list): List of detected objects with:
                    - name (str): Object identifier
                    - angle (float): Orientation angle in degrees
                    - center (tuple): Center coordinates (x, y)
                - visualization_path (str): Path to saved visualization
        """
        try:
            # Validate image
            if img is None or not isinstance(img, np.ndarray):
                return {
                    'success': False,
                    'message': 'Invalid image data',
                    'objects': [],
                    'visualization_path': None
                }
            
            # Run YOLO prediction
            pred_polys = yolo_predict_polygons(img)
            polygons = [p[1] for p in pred_polys]
            
            if not polygons:
                return {
                    'success': True,
                    'message': 'No objects detected',
                    'objects': [],
                    'visualization_path': None
                }
            
            # Prepare image data
            image_data = {
                'name': image_name,
                'image': img,
                'polygons': polygons
            }
            
            # Analyze regions
            vis = self.analyzer.analyze(image_data)
            
            # Extract object information
            objects = []
            for poly_idx, poly in enumerate(polygons, start=1):
                pts = np.array(poly, dtype=np.int32)
                para = self.analyzer.approximate_parallelogram(pts)
                center = self.analyzer.compute_center(para)
                p1, p2, angle = self.analyzer.longer_side_and_angle(para)
                
                objects.append({
                    'name': f'object_{poly_idx}',
                    'angle': float(angle),
                    'center': center
                })
            
            # Save visualization
            visualization_path = None
            if self.analyzer.store_result:
                visualization_path = os.path.join(
                    self.analyzer.save_dir, 
                    f"{image_name}_analysis.png"
                )
                cv2.imwrite(visualization_path, vis)
            
            return {
                'success': True,
                'message': f'Successfully processed {len(objects)} objects',
                'objects': objects,
                'visualization_path': visualization_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing frame: {str(e)}',
                'objects': [],
                'visualization_path': None
            }
    
    def process_image(self, image_path):
        """
        Process an image using the box model.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            dict: Dictionary containing:
                - success (bool): Whether processing was successful
                - message (str): Status message
                - objects (list): List of detected objects with:
                    - name (str): Object identifier
                    - angle (float): Orientation angle in degrees
                    - center (tuple): Center coordinates (x, y)
                - visualization_path (str): Path to saved visualization
        """
        try:
            # Load image
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'message': f'Image not found: {image_path}',
                    'objects': [],
                    'visualization_path': None
                }
            
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'success': False,
                    'message': f'Failed to read image: {image_path}',
                    'objects': [],
                    'visualization_path': None
                }
            
            # Get image name
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Run YOLO prediction
            pred_polys = yolo_predict_polygons(img)
            polygons = [p[1] for p in pred_polys]
            
            if not polygons:
                return {
                    'success': True,
                    'message': 'No objects detected',
                    'objects': [],
                    'visualization_path': None
                }
            
            # Prepare image data
            image_data = {
                'name': image_name,
                'image': img,
                'polygons': polygons
            }
            
            # Analyze regions
            vis = self.analyzer.analyze(image_data)
            
            # Extract object information
            objects = []
            for poly_idx, poly in enumerate(polygons, start=1):
                pts = np.array(poly, dtype=np.int32)
                para = self.analyzer.approximate_parallelogram(pts)
                center = self.analyzer.compute_center(para)
                p1, p2, angle = self.analyzer.longer_side_and_angle(para)
                
                objects.append({
                    'name': f'object_{poly_idx}',
                    'angle': float(angle),
                    'center': center
                })
            
            # Save visualization
            visualization_path = None
            if self.analyzer.store_result:
                visualization_path = os.path.join(
                    self.analyzer.save_dir, 
                    f"{image_name}_analysis.png"
                )
                cv2.imwrite(visualization_path, vis)
            
            return {
                'success': True,
                'message': f'Successfully processed {len(objects)} objects',
                'objects': objects,
                'visualization_path': visualization_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing image: {str(e)}',
                'objects': [],
                'visualization_path': None
            }

