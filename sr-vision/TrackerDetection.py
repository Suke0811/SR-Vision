from FrameHandler import FrameHandler
from IntelRealsenseHandler import IntelRealsenseHandler
from DetectionHandler import DetectionHander
import numpy as np
import cv2
import traceback

class TrackerDetection:
    
    def __init__(self, model_path, log=False, display=False):
        self.detector = DetectionHander(model_path, log, display)
        self.camera = IntelRealsenseHandler()
        # this is used inside frame handler
        self._classes = []
        self.frame_handler = FrameHandler(self.camera, self._classes)
        self.positions = np.empty((0, 4), dtype=np.float32)
        self.display = display
        print(f"DISPLAY IS {self.display}")
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
                print('HITTING EXCEPTION IN TRACKER')
                print(f"Error: {e}")
                print(traceback.format_exc())
                
                # self.run = False
    
    def stop (self):
        self.run = False

    def _display_frame(self, display, color_frame, bboxes):
        if display:
            display_frame = self.frame_handler.display_data(color_frame, bboxes)
            cv2.imshow('Detection Inference', display_frame)
            cv2.waitKey(1)
    
    def update(self):
        """
        Updates object state by retrieving frames, performing detection, and displaying if set to True.
        """
        depth_frame, color_frame = self.camera.get_frames()
        bboxes = self.detector.detection(color_frame)
        self.positions = self.frame_handler.get_xyz(depth_frame, bboxes)
        self._display_frame(self.display, color_frame, bboxes)

            
    @property
    def classes(self):
        return self._classes
    
    @classes.setter
    def classes(self, classes):
        self._classes = classes