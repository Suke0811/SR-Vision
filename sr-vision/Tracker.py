from Base import FrameHandler
from Base import IntelRealsenseHandler
from Base import SegmentationHandler
from numpy import np
import cv2

class Tracker:
    def __init__(self, model_path, log=False, display=True):
        self.segmenter = SegmentationHandler(model_path, log, display)
        self.camera = IntelRealsenseHandler()
        self.frame_handler = FrameHandler(self.camera)
        self.positions = np.empty((0, 4), dtype=np.float32)
        self.display = display
        self.log = log
        self.run = True

    def run(self):
        while self.run:
            try:
                self.update()
            except:
                self.run = False
    
    def stop (self):
        self.run = False

    def _dispaly_frame(self, display, color_frame, polygons, bboxes):
        if display:
            display_frame = self.frame_handler.display_data(color_frame, polygons, bboxes)
            cv2.imshow('Segmentation Inference', display_frame)
            cv2.waitKey(1)
    
    def update(self):
        """
        Updates object state by retrieving frames, performing segmentation, and displaying if set to True.
        """
        depth_frame, color_frame = self.camera.get_frames()
        bboxes, polygons = self.segmenter.segmentation(color_frame)
        self.positions = self.frame_handler.get_xyz(depth_frame, polygons)
        self._dispaly_frame(self.display, color_frame, polygons, bboxes)
            