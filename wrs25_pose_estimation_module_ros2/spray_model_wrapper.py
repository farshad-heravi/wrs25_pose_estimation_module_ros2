"""
Wrapper module for the Spray Vision model.
Interfaces with the spray model from WRS-Vision folder.
"""

import sys
import os
import cv2
import numpy as np

# Get WRS-Vision path from environment or use default
WRS_VISION_ROOT = '/home/mpt/projects/wrs_vision/WRS-Vision'
# WRS_VISION_ROOT = os.environ.get('WRS_VISION_ROOT', 
#                                   os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'WRS-Vision'))
SPRAY_MODEL_PATH = os.path.join(WRS_VISION_ROOT, 'spray')
SPRAY_MODEL_PATH = os.path.abspath(SPRAY_MODEL_PATH)

print(f"WRS_VISION_ROOT: {WRS_VISION_ROOT}")
print(f"SPRAY_MODEL_PATH: {SPRAY_MODEL_PATH}")

if not os.path.exists(SPRAY_MODEL_PATH):
    raise RuntimeError(
        f"Spray model not found at: {SPRAY_MODEL_PATH}\n"
        f"Please set WRS_VISION_ROOT environment variable to point to the WRS-Vision directory.\n"
        f"Example: export WRS_VISION_ROOT=/home/mpt/projects/wrs_vision/WRS-Vision"
    )

sys.path.insert(0, SPRAY_MODEL_PATH)

# Set the PROJECT_ROOT in config before importing
import config as spray_config
spray_config.PROJECT_ROOT = SPRAY_MODEL_PATH
spray_config.IMG_DIR = os.path.join(SPRAY_MODEL_PATH, "test_images")
spray_config.OUT_DIR = os.path.join(SPRAY_MODEL_PATH, "results")
spray_config.YOLO_MODEL_PATH = os.path.join(SPRAY_MODEL_PATH, "weights", "best.pt")

from image_io import yolo_predict_polygons
from preprocess import process_image_data
from mask_generator import MaskGenerator
from region_analysis import RegionAnalyzer


class SprayModelWrapper:
    """Wrapper class for the Spray Vision model."""
    
    def __init__(self):
        """Initialize the spray model wrapper."""
        preprocess_dir = os.path.join(spray_config.OUT_DIR, "preprocess_result")
        self.mask_gen = MaskGenerator(save_dir=os.path.join(spray_config.OUT_DIR, "mask_result"))
        self.analyzer = RegionAnalyzer(save_dir=os.path.join(spray_config.OUT_DIR, "final_result"))
        print("Spray model wrapper initialized")
    
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
                    - object_id (int): Object identifier
                    - tail (list): Tail coordinates [x, y]
                    - head (list): Head coordinates [x, y]
                    - centroid (list): Centroid coordinates [x, y]
                    - orientation_deg (float): Orientation angle in degrees
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
            
            # Prepare image data
            image_data = {
                'name': image_name,
                'image': img,
                'pred': pred_polys
            }
            
            # Step 2: Preprocess predicted polygons
            preprocess_dir = os.path.join(spray_config.OUT_DIR, "preprocess_result")
            processed = process_image_data(image_data, preprocess_dir)
            
            # Step 3: Generate mask from processed polygons
            mask_result = self.mask_gen.generate({
                'name': processed['name'],
                'image': image_data['image'],
                'polygons': processed['polygons']
            })
            
            # Step 4: Analyze the final mask
            analysis_result = self.analyzer.analyze({
                'name': processed['name'],
                'image': image_data['image'],
                'mask': mask_result['mask_gap_filled']
            })
            
            if analysis_result is None:
                return {
                    'success': False,
                    'message': 'Analysis failed - no valid regions detected',
                    'objects': [],
                    'visualization_path': None
                }
            
            # Extract results
            objects = analysis_result.get('results', [])

            return {
                'success': True,
                'message': f'Successfully processed {len(objects)} objects',
                'objects': objects,
                'visualization_path': None
                # 'visualization_path': visualization_path
            }
            
            # # Get visualization path
            # visualization_path = None
            # if self.analyzer.store_result and self.analyzer.save_dir:
            #     visualization_path = os.path.join(
            #         self.analyzer.save_dir,
            #         f"{image_name}_head_tail.png"
            #     )
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'message': f'Error processing frame: {str(e)}\n{traceback.format_exc()}',
                'objects': [],
                'visualization_path': None
            }

    def illustrate_result(self, img, objects, select_bottle=None, valid_area=None):
        """Illustrate the result."""
        if img is None or not isinstance(img, np.ndarray):
            return
        
        if valid_area is not None:
            # draw a rectangle on the image
            cv2.rectangle(img, (valid_area[0], valid_area[1]), (valid_area[2], valid_area[3]), (0, 0, 255), 2)
        
        # illustrate the all bottles
        for obj in objects:
            # Draw line connecting tail to head
            cv2.line(img, (obj['tail'][0], obj['tail'][1]), (obj['head'][0], obj['head'][1]), (0, 0, 255), 2)
            # Draw circles: head (red), centroid (red), tail (blue)
            cv2.circle(img, (obj['head'][0], obj['head'][1]), 5, (0, 0, 255), -1)  # Red for head
            cv2.circle(img, (obj['centroid'][0], obj['centroid'][1]), 5, (0, 0, 255), -1)  # Red for centroid
            cv2.circle(img, (obj['tail'][0], obj['tail'][1]), 5, (255, 0, 0), -1)  # Blue for tail

        # illustrate the selected bottle
        if select_bottle is not None:
            # Draw line connecting tail to head
            cv2.line(img, (select_bottle['tail'][0], select_bottle['tail'][1]), (select_bottle['head'][0], select_bottle['head'][1]), (0, 255, 0), 4)
            # Draw circles: head (red), centroid (green for selected), tail (blue)
            cv2.circle(img, (select_bottle['head'][0], select_bottle['head'][1]), 5, (0, 0, 255), -1)  # Red for head
            cv2.circle(img, (select_bottle['centroid'][0], select_bottle['centroid'][1]), 5, (0, 255, 0), -1)  # Green for centroid
            cv2.circle(img, (select_bottle['tail'][0], select_bottle['tail'][1]), 5, (255, 0, 0), -1)  # Blue for tail

        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_image(self, image_path):
        """
        Process an image using the spray model.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            dict: Dictionary containing:
                - success (bool): Whether processing was successful
                - message (str): Status message
                - objects (list): List of detected objects with:
                    - object_id (int): Object identifier
                    - tail (list): Tail coordinates [x, y]
                    - head (list): Head coordinates [x, y]
                    - centroid (list): Centroid coordinates [x, y]
                    - orientation_deg (float): Orientation angle in degrees
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
            
            # Prepare image data (similar to load_image_data but from file path)
            image_data = {
                'name': image_name,
                'image': img,
                'pred': pred_polys
            }
            
            # Step 2: Preprocess predicted polygons
            preprocess_dir = os.path.join(spray_config.OUT_DIR, "preprocess_result")
            processed = process_image_data(image_data, preprocess_dir)
            
            # Step 3: Generate mask from processed polygons
            mask_result = self.mask_gen.generate({
                'name': processed['name'],
                'image': image_data['image'],
                'polygons': processed['polygons']
            })
            
            # Step 4: Analyze the final mask
            analysis_result = self.analyzer.analyze({
                'name': processed['name'],
                'image': image_data['image'],
                'mask': mask_result['mask_gap_filled']
            })
            
            if analysis_result is None:
                return {
                    'success': False,
                    'message': 'Analysis failed - no valid regions detected',
                    'objects': [],
                    'visualization_path': None
                }
            
            # Extract results
            objects = analysis_result.get('results', [])
            
            # Get visualization path
            visualization_path = None
            if self.analyzer.store_result and self.analyzer.save_dir:
                visualization_path = os.path.join(
                    self.analyzer.save_dir,
                    f"{image_name}_head_tail.png"
                )
            
            # Display the original image and detection results
            cv2.imshow('Original Image', img)
            if visualization_path and os.path.exists(visualization_path):
                result_img = cv2.imread(visualization_path)
                if result_img is not None:
                    cv2.imshow('Detection Results', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return {
                'success': True,
                'message': f'Successfully processed {len(objects)} objects',
                'objects': objects,
                'visualization_path': visualization_path
            }
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'message': f'Error processing image: {str(e)}\n{traceback.format_exc()}',
                'objects': [],
                'visualization_path': None
            }

