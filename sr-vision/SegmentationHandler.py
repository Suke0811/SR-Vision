from Base import SegmentationHandlerBase
import cv2
import pyrealsense2
import numpy as np
import pathlib as Path
import torch
import time
from ultralytics import YOLO


class SegmentationHandler(SegmentationHandlerBase):
    def __init__(self, model_path, log, display):
        # init model variables
        self.base_dir = Path(__file__).resolve().parent.parent.parent
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
        
        
    def get_depth_at_centroid_seg(self, polygon):
        # Calculate the centroid
        M = cv2.moments(np.array(polygon, dtype=np.int32))
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            return None, 0, 0

        # Get depth value at centroid
        error, point3D = self.point_cloud.get_value(center_x, center_y)
        if self.err == sl.ERROR_CODE.SUCCESS:
            x = point3D[0]
            y = point3D[1]
            z = point3D[2]
            color = point3D[3]
            return z, x, -y, center_x, center_y
        else:
            return None, 0, 0, 0, 0
    
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
       
