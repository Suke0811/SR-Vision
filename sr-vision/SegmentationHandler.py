from Base import SegmentationHandlerBase
import cv2
import pyrealsense2 as rs
import numpy as np
import pathlib as Path
import torch
import time
from ultralytics import YOLO


class SegmentationHandler(SegmentationHandlerBase):
    def __init__(self, model_path, log=False, display=False, max_model_size=640, det_conf=0.1, *args, **kwargs):
        # init model variables
        self.base_dir = Path(__file__).resolve().parent
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.classes = ['Door Handle', 'Door Knob']
        self.results = None
        self.max_model_size = max_model_size
        self.det_conf = det_conf
        
        # list of positions based on inference results
        self.positions = np.empty((0, 4), dtype=np.float32)
        
        # init camera variables
        self.frame = None
        self.depth_frame = None
        
        # segmentation flags
        self.log = log
        self.display = display
        
        # init processing variables
        self.bboxes = None # contains (classification ID, bbox coordinates)
        self.masks = None
        
    '''Getters:'''
    def get_bboxes(self):
        return self.bboxes
    
    def get_masks(self):
        return self.masks
    
    def get_positions(self):
        return self.positions
    
    def get_frame(self):
        return self.frame
        
    '''Setters:'''

    def set_frame(self, frame):
        self.frame = frame
        
    def set_depth_frame(self, depth_frame):
        self.depth_frame = depth_frame
        
    def set_display(self, display):
        self.display = display
    
    '''Processers:'''
    
    def process_model(self):
        # run model to get results
        self.results = self.model(self.frame, imgsz=(self.max_model_size), stream=True, conf=self.det_conf)
        # reset bboxes and masks list
        self.bboxes = None
        self.masks = None
    
    # Main function to run the model
    def segmentation(self):
        # run inference on frame
        self.process_model()
        
        result = None
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
                
                self.bboxes.append(cls, bbox)
                
                # Extract segmentation mask
                polygon = mask.xy[0]
                
                self.masks.append(polygon)
                
                
                
                
                
                
                
                
                
                
        
        

