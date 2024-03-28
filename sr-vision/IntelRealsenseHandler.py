from .Base import DEFAULT_TIMEOUT
from Base import IntelRealsenseHandlerBase
import pyrealsense2 as rs
import numpy as np


class IntelRealsenseHandler(IntelRealsenseHandlerBase):
    pass

    def __init__(self, timeout=DEFAULT_TIMEOUT):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Align the two cameras since there is physical offset
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        
        # Variable to hold camera intrinsics
        self.intrinsics = None
        
        # np arrays for both depth and color images
        self.depth_image = None
        self.color_image = None
        
    def start_camera(self):
        pass
    
    def stop_camera(self):
        pass
    
    def get_frames(self, wait=True):
        try:
            # Attempt to retrieve the next set of frames
            if wait:
                frames = self.pipeline.poll_for_frames()
            else:
                frames = self.pipeline.wait_for_frames()
                
            # Check if frames are available
            if frames:
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    return None, None

                # Convert color image to numpy array
                color_image = np.asanyarray(color_frame.get_data())

                return aligned_depth_frame, color_image
            else:
                # No new frames available
                return None, None
        
        except RuntimeError:
            pass
        except KeyboardInterrupt:
            self.stop_camera()
        