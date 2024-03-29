from Base import FrameHandler
from Base import IntelRealsenseHandler
from Base import SegmentationHandler
from numpy import np
import cv2

class Tracker:
    def __init__(self, model_path, log=False, display=True):
        """
        """
        self.segmenter = SegmentationHandler(model_path, log, display)
        self.camera = IntelRealsenseHandler()
        self.frame_handler = FrameHandler()
        self.positions = np.empty((0, 4), dtype=np.float32)
        self.display = display
        self.log = log

    def run(self):
        """
        """
        pass
    
    def update(self):
        """
        """
        depth_frame, color_frame = self.camera.get_frames
        bboxes, polygons = self.segmenter.segmentation(color_frame)
        self.positions = self.frame_handler.get_positions(depth_frame, polygons)
        if self.display:
            display_frame = self.frame_handler.display_data(color_frame, polygons, bboxes)
            cv2.imshow('YOLOv8 Inference', display_frame)
            cv2.waitKey(1)
            