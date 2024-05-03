from Base import SegmentationHandlerBase
import cv2
import numpy as np
from pathlib import Path
import torch
import time
from ultralytics import YOLO

class SegmentationHandler(SegmentationHandlerBase):
    def __init__(self, model_path, log=False, display=False, max_model_size=640, det_conf=0.2, iou=0.6, *args, **kwargs):
        # init model variables
        self.base_dir = Path(__file__).resolve().parent
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.results = None
        self.max_model_size = max_model_size
        self.det_conf = det_conf
        self.iou = iou

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
        self.polygons = []

    '''Getters:'''
    # @property
    # def bboxes(self):
    #     return self.bboxes

    # @property
    # def polygons(self):
    #     return self.polygons

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
        self.results = self.model(frame, imgsz=(self.max_model_size), stream=True, conf=self.det_conf, iou=self.iou, verbose=self._log, half=False, device='cpu') 

        # reset bboxes and polygons list
        self.bboxes = []
        self.polygons = []

    def segmentation(self, frame):
        """
        Perform segmentation on the input frame and extract bounding boxes and segmentation masks.
        :param frame: The input frame for segmentation
        :return: A tuple containing the bounding boxes and segmentation masks
        """
        # run inference on frame
        self._process_model(frame)
        result = None

        # extract the single inference from results
        if self.results is not None:
            for inference in self.results:
                result = inference
                if result.boxes and result.masks:
                    for box, mask in zip(result.boxes, result.masks):
                        # Extract bounding box classification
                        cls_ = int(box.cls[0].item())
                        # Extract segmentation mask
                        polygon = mask.xy[0]

                        # Extract bounding box
                        bbox = box.xyxy[0].cpu().numpy().astype(int)

                        # Extract confidence
                        confidence = box.conf[0]

                        # make bbox and polgyon tuple
                        bbox_tuple = (cls_, confidence, bbox)
                        polgyon_tuple = (cls_, confidence, polygon)

                        self.polygons.append(polgyon_tuple)
                        self.bboxes.append(bbox_tuple)

        return self.bboxes, self.polygons