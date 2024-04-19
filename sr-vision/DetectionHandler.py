import cv2
import numpy as np
from pathlib import Path
import torch
import time
from ultralytics import YOLO

class DetectionHander():
    def __init__(self, model_path, log=False, display=False, max_model_size=640, det_conf=0.2, *args, **kwargs):
        # init model variables
        self.base_dir = Path(__file__).resolve().parent
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.results = None
        self.max_model_size = max_model_size
        self.det_conf = det_conf

        # list of positions based on inference results
        self._positions = np.empty((0, 4), dtype=np.float32)

        # init camera variables
        self._frame = None
        self._depth_frame = None

        # segmentation flags
        self._log = log
        self._display = display

        # init processing variables
        self.bboxes = [] # contains (classification ID, bbox coordinates)

    '''Getters:'''
    @property
    # def bboxes(self):
    #     return self.bboxes

    @property
    def frame(self):
        return self._frame

    @property
    def depth_frame(self):
        return self._depth_frame

    @property
    def display(self):
        return self._display

    '''Setters:'''
    @frame.setter
    def frame(self, value):
        self._frame = value

    @depth_frame.setter
    def depth_frame(self, value):
        self._depth_frame = value

    @display.setter
    def display(self, value):
        self._display = value

    '''Processors:'''
    def _process_model(self, frame):
        # run model to get results
        self.results = self.model(frame, imgsz=(self.max_model_size), stream=True, conf=self.det_conf)

        # reset bboxes 
        self.bboxes = []

    def detection(self, frame):
        """
        Perform object detection on a given frame.

        Parameters:
        - frame: the frame to run inference on

        Returns:
        - bboxes: list of bounding boxes for detected objects
        """
        # run inference on frame
        self._process_model(frame)
        result = None

        # extract the single inference from results
        if self.results is not None:
            for inference in self.results:
                result = inference
                if result.boxes:
                    for box in result.boxes:
                        # Extract bounding box classification
                        cls_ = int(box.cls[0].item())

                        # Extract bounding box
                        bbox = box.xyxy[0].cpu().numpy().astype(int)

                        # Extract confidence
                        confidence = box.conf[0]
                        bbox = (cls_, confidence, bbox)
                        self.bboxes.append(bbox)

        return self.bboxes