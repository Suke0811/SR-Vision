from Base import SegmentationHandlerBase
import cv2
import pyrealsense2 as rs
import numpy as np
import pathlib as Path
import torch
import time
from ultralytics import YOLO


class SegmentationHandler(SegmentationHandlerBase):
    def __init__(self, model_path, log, display):
        # init model variables
        self.base_dir = Path(__file__).resolve().parent
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.classes = ['Door Handle', 'Door Knob']
        self.results = None
        
        # init camera variables
        self.frame = None
        self.point_cloud = None
        
        # segmentation flags
        self.log = log
        self.display = display
        
        # init processing variables
        self.bbox = None
        self.mask = None
        
    '''Getters:'''
    def get_bbox(self):
        return self.bbox
    
    def get_mask(self):
        return self.mask
    
    def get_positions(self):
        return self.positions
    
    def get_frame(self):
        return self.frame
     
    def get_depth_at_centroid_seg(self, polygon):
        """
        Calculate the centroid of the given polygon and retrieve the depth value at the centroid. 

        Parameters:
        polygon (list): List of points representing the polygon.

        Returns:
        float or None: Depth value at the centroid, or None if no depth value is available.
        float: X coordinate of the centroid in camera coordinates.
        float: Y coordinate of the centroid in camera coordinates.
        int: X pixel coordinate of the centroid in the depth frame.
        int: Y pixel coordinate of the centroid in the depth frame.
        """
        # Calculate the centroid
        M = cv2.moments(np.array(polygon, dtype=np.int32))
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            return None, 0, 0

        # Get depth value at centroid
        depth_frame = self.realsense_handler.get_depth_frame()
        if depth_frame:
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_pixel = [center_x, center_y]
            depth_in_meters = depth_frame.get_distance(depth_pixel[0], depth_pixel[1])

            if depth_in_meters > 0:
                # Convert depth pixel to 3D point in camera coordinates
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, depth_pixel, depth_in_meters)
                x = depth_point[0]
                y = depth_point[1]
                z = depth_point[2]
                return z, -x, -y, center_x, center_y
            else:
                return None, 0, 0, 0, 0
        else:
            return None, 0, 0, 0, 0
        
    '''Setters:'''

    def set_frame(self, frame):
        self.frame = frame
        
    def set_point_cloud(self, point_cloud):
        self.point_cloud = point_cloud
        
    def set_display(self, display):
        self.display = display
    
    '''Display Functions:'''
 
    def display_data(self, box, polygon, bbox, ID, center_x, center_y):
        """
        A function to display data including bounding box, segmentation polygon, and label on the frame.

        Parameters:
        - box: The bounding box object
        - polygon: The segmentation polygon points
        - bbox: The bounding box coordinates
        - ID: The class ID
        - center_x: The x-coordinate of the centroid
        - center_y: The y-coordinate of the centroid

        Returns:
        This function applies the changes to the frame and does not return anything.
        """
        # get confidence and class name
        confidence = box.conf[0]
        current_class_name = self.classes[ID]
        
        # Draw bounding box 
        x1, y1, x2, y2 = bbox
        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        xyz_label = f"X: {self.center_3d[0]:.2f}, Y: {self.center_3d[1]:.2f}, Z: {self.center_3d[2]:.2f}"
        label = f"{current_class_name} {confidence:.2f} ({xyz_label})"
        
        # Draw segmentation polygon
        cv2.polylines(self.frame, [np.array(polygon, dtype=np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)
        
        # Draw a circle at the 2D centroid
        cv2.circle(self.frame, (center_x, center_y), 5, (0, 0, 255), -1)
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Adjust the y position of the label to sit on top or inside the bounding box
        # label_y_position = max(y1, label_height + baseline) 
        label_y_position = y1 - label_height + baseline + 4
        box_y_position = y1 - label_height - baseline 
        if (box_y_position < 0): 
            label_y_position = y1 + label_height - baseline + 6 
            box_y_position = y1 + label_height + baseline                      

        # Create a rectangle for the label background and put the text directly on the bounding box
        cv2.rectangle(self.frame, (x1, box_y_position), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(self.frame, label, (x1, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    '''Processers:'''
    
    def process_model(self):
        self.results = self.model(self.frame, imgsz=(self.max_model_size), stream=True, conf=self.det_conf)
    
    # Main function to run the model
    def segmentation(self):
        # run inference on frame
        self.process_model()
        # initialize 2D Matrix for positions of detected objects
        self.positions = np.empty((0, 4), dtype=np.float32)
        
        # extract the single inference from results
        if self.results is not None:
            for inference in self.results:
                result = inference
                
        if result.boxes and result.masks:
            for box, mask in zip(result.boxes, result.masks):
                # Extract bounding box
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                # Extract bounding box classification
                cls = int(box.cls[0].item())
                
                # Extract segmentation mask
                polygon = mask.xy[0]
                
                # Extract depth at centroid
                depth, x, y, cx, cy = self.get_depth_at_centroid_seg(polygon)
                position = np.array([[cls, x, y, depth]])
                # Append position of detection to 2D Matrix
                self.positions = np.vstack((self.positions, position))
                
                if (self.display):
                    # display data on frame
                    self.display_data(box, polygon, bbox, cls, center_x, center_y)
                
                
                
                
                
                
                
                
        
        
        
# ------------------------------------------------ 
        process_model_time = time.time()  # Capture start time
        self.process_model()
        # refresh positions every cycle
        self.positions = []
        # depth_image = self.depth_seg()
        process_model_end_time = time.time()  # Capture end time
        process_model_elapsed_time = process_model_end_time - process_model_time  # Calculate elapsed time

        elapsed_time1 = 0
        elapsed_time0 = 0

        if self.results is not None:
            
            for_time = time.time()
            for f in self.results:
                for_end = time.time()
                for_elapsed = for_end - for_time # Calculate elapsed time
                result = f

            start_time0 = time.time()
            if result.boxes:
                start_time1 = time.time()
                for box in result.boxes:
                    # Extract bounding box
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0].item())

                    # get ball 3D pose estimation
                    depth, x, y, center_x, center_y = self.get_pose_at_centroid_bbox(bbox)
                    self.center_3d = [x, y, depth]

                    if cls == 0:
                        if not self.ball_checker(bbox, depth):
                            continue
                        else:
                            position = (cls, x, y, depth)
                            self.positions.append(position)
                    else:
                        position = (cls, x, y, depth)
                        self.positions.append(position)
                    
                    if (self.display):
                        # display data on frame
                        self.display_data(box, bbox, cls, center_x, center_y)
                end_time1 = time.time()  # Capture end time
                elapsed_time1 = end_time1 - start_time1 # Calculate elapsed time
            end_time0 = time.time()  # Capture end time
            elapsed_time0 = end_time0 - start_time0 # Calculate elapsed time
            if (self.log):
                print(f"for dt: {for_elapsed}")
                print(f"process_model dt: {process_model_elapsed_time}")
                print(f"results dt: {elapsed_time0}")
                print(f"run_model dt (just one box): {elapsed_time1}")

        if (self.display):
            cv2.imshow('YOLOv8 Inference', self.frame)
            # cv2.imshow('Depth Sobel Image', depth_image)
            cv2.waitKey(1)
       
