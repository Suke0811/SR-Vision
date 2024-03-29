from Base import FrameHandlerBase
from Base import IntelRealsenseHandler
from Base import SegmentationHandler
from numpy import np

class Tracker:
    def __init__(self):
        """
        """
        self.segmenter = SegmentationHandler(model_path, log, display)
        self.camera = IntelRealsenseHandler()
        self.positions = np.empty((0, 4), dtype=np.float32)
        pass

    def run(self):
        """
        """
        pass
    
    def update(self):
        """
        """
        pass