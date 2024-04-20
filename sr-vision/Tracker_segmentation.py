from FrameHandler import FrameHandler
from IntelRealsenseHandler import IntelRealsenseHandler
from SegmentationHandler import SegmentationHandler
import numpy as np
import cv2
import traceback

class Tracker:
    def __init__(self, model_path, log=False, display=True):
        self.segmenter = SegmentationHandler(model_path, log, display)
        self.camera = IntelRealsenseHandler()
        # this is used inside frame handler
        self.classes_ = []
        self.colors_ = {}
        self.frame_handler = FrameHandler(self.camera, self._classes)
        self.positions = np.empty((0, 4), dtype=np.float32)
        self.display = display
        print(f"Display is {self.display}")
        self.log = log
        self.run = True
    
    def __del__(self):
        self.uninit()

    def uninit(self):
        self.stop()
        self.camera.stop_camera()
        cv2.destroyAllWindows()
        
    def run_model(self):
        self.camera.start_camera()
        while self.run:
            try:
                self.update()

            except Exception as e:
                print('HITTING EXCEPTION')
                print(f"Error: {e}")
                print(traceback.format_exc())
                # self.run = False
    
    def stop (self):
        self.run = False

    def _dispaly_frame(self, display, color_frame, bboxes, polygons):
        if display:
            display_frame = self.frame_handler.display_data(color_frame, bboxes, polygons)
            cv2.imshow('Segmentation Inference', display_frame)
            cv2.waitKey(1)
    
    def update(self):
        """
        Updates object state by retrieving frames, performing segmentation, and displaying if set to True.
        """
        depth_frame, color_frame = self.camera.get_frames(wait=False)
        bboxes, polygons = self.segmenter.segmentation(color_frame)
        self.positions = self.frame_handler.get_xyz(depth_frame, polygons)
        self._dispaly_frame(self.display, color_frame, bboxes, polygons)

    @property
    def classes(self):
        return self.classes_
    
    @classes.setter
    def classes(self, classes):
        self.classes_ = classes
    
    @property
    def colors(self):
        return self.colors_
    
    @colors.setter
    def colors(self, colors):
        self.colors_ = colors