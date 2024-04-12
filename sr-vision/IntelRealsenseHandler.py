from Base import DEFAULT_TIMEOUT
from Base import IntelRealsenseHandlerBase
import pyrealsense2 as rs
import numpy as np
import cv2


class IntelRealsenseHandler(IntelRealsenseHandlerBase):
    def __init__(self, timeout=DEFAULT_TIMEOUT):
        super().__init__(timeout) # init params in base class
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Align the two cameras since there is physical offset
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        
    def start_camera(self):
        # Start the pipeline
        self.profile = self.pipeline.start(self.config)
        # Get the camera intrinsics from the color stream
        self._intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    
    def stop_camera(self):
        # Stop the pipeline
        self.pipeline.stop()
    
    '''Getters:'''
    
    @property
    def color_frame(self):
        return self.color_frame
    
    @property
    def depth_frame(self):
        return self._depth_frame
    
    @property
    def intrinsics(self):
        return self._intrinsics
    
    '''Setters:'''
    @color_frame.setter
    def color_frame(self, value):
        self._color_frame = value
    
    @depth_frame.setter
    def depth_frame(self, value):
        self._depth_frame = value

    @intrinsics.setter
    def intrinsics(self, value):
        self._intrinsics = value
    
    def get_frames(self, wait=None):
        # Use provided wait value or fall back to class default
        if wait is None:
            wait = self.wait
        try:
            # Retrieve next set of frames
            frames = self.pipeline.wait_for_frames() if wait else self.pipeline.poll_for_frames()
            
            # Align and retrieve depth and color frames
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            # Check if all frames are available
            if aligned_depth_frame and color_frame:
                # Convert color image to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                return aligned_depth_frame, color_image
            else:
                return None, None
        
        except RuntimeError:
            pass

    
    def get_3D_pose(self, depth_frame, polygon):
        """
        Calculate the centroid of the given polygon and retrieve the depth value at the centroid. 
 
        Parameters:
        depth_frame (rs.frame): Depth frame from the Intel Realsense camera.
        polygons (list): List of points representing the polygon.

        Returns:
        float or None: Depth value at the centroid, or None if no depth value is available.
        float: X coordinate of the centroid in camera coordinates.
        float: Y coordinate of the centroid in camera coordinates.
        int: X pixel coordinate of the centroid in the depth frame.
        int: Y pixel coordinate of the centroid in the depth frame.
        """
        center_x, center_y = self._get_centroid_pixel(polygon)
        if center_x is not None and center_y is not None:
            depth, x, y = self._get_depth_at_centroid(depth_frame, center_x, center_y)    
            return depth, x, y, center_x, center_y
        else:
            return None, 0, 0, 0, 0
        
    def _get_centroid_pixel(self, polygon):
        # Calculate the centroid
        M = cv2.moments(np.array(polygon, dtype=np.int32))
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return center_x, center_y
        else:
            return None, None
        
    def _get_depth_at_centroid(self, depth_frame, center_x, center_y):
        zxy_pose = None, None, None
        # Get depth value at centroid
        if depth_frame:
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_pixel = [center_x, center_y]
            depth_in_meters = depth_frame.get_distance(depth_pixel[0], depth_pixel[1])

            if depth_in_meters > 0:
                # Convert depth pixel to 3D point in camera coordinates
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, depth_pixel, depth_in_meters)
                x = depth_point[0]
                y = depth_point[1]
                z = depth_point[2]
                # flip y so up is positive
                # z = z - 0.0042 # slight camera offset for camera to lens protector
                zxy_pose = z, x, -y
        # Return the depth, x, and y coordinates
        return zxy_pose
        
    
    
