from FrameHandler import FrameHandler
from IntelRealsenseHandler import IntelRealsenseHandler
from SegmentationHandler import SegmentationHandler
import numpy as np
import cv2
import traceback
import time

class TrackerSegmentation:
    def __init__(self, model_path, classes=None, colors=None, log=False, display=True, max_model_size=640, det_conf=0.2, iou=0.6, *args, **kwargs):
        self.segmenter = SegmentationHandler(model_path=model_path,
                                             log=log, 
                                             display=display,
                                             max_model_size=max_model_size,
                                             det_conf=det_conf,
                                             iou=iou)
        self.camera = IntelRealsenseHandler()
        # This is used inside frame handler
        self._classes = classes if classes is not None else []
        self._colors = colors if colors is not None else {}
        self.frame_handler = FrameHandler(self.camera, self._classes, self._colors)
        self.positions = np.empty((0, 4), dtype=np.float32)
        self.display = display
        self.display_frame = None
        print(f"Display is {self.display}")
        self.log = log
        self.run = True
    
    def __del__(self):
        self.uninit()

    def uninit(self):
        self.stop()
        self.camera.stop_camera()
        cv2.destroyAllWindows()
        
    
    def stop (self):
        self.run = False

    def _display_frame(self, color_frame, bboxes, polygons):
        if self.display:
            self.display_frame = self.frame_handler.display_data(color_frame, bboxes, polygons)
            cv2.imshow('Segmentation Inference', self.display_frame)
            cv2.waitKey(1)
    
    def start_camera(self):
        self.camera.start_camera()
    
    def update(self):
        """
        Updates object state by retrieving frames, performing segmentation, and displaying if set to True.
        """
        depth_frame, color_frame = self.camera.get_frames(wait=True)
        # Check if depth and color frames are available
        if depth_frame is not None and color_frame is not None:    
            bboxes, polygons = self.segmenter.segmentation(color_frame)
            self.positions = self.frame_handler.get_xyz(depth_frame, polygons)
            self._display_frame( color_frame, bboxes, polygons)

    def run_model(self):
        self.camera.start_camera()
        while self.run:
            try:
                start_time = time.perf_counter()
                self.update()
                end_time = time.perf_counter()
                if self.log:
                    print(f"Time taken: {end_time - start_time}")

            except Exception as e:
                print('HITTING EXCEPTION')
                print(f"Error: {e}")
                print(traceback.format_exc())
                # self.run = False
                pass

    @property
    def classes(self):
        return self._classes
    
    @classes.setter
    def classes(self, classes):
        self._classes = classes
    
    @property
    def colors(self):
        return self._colors
    
    @colors.setter
    def colors(self, colors):
        self._colors = colors

    def get_positions(self):
        return self.positions
    
    def get_display_frame(self):
        return self.display_frame