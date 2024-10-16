from Base import DEFAULT_TIMEOUT
from Base import IntelRealsenseHandlerBase
import pyrealsense2 as rs
import numpy as np
import cv2


class IntelRealsenseHandler(IntelRealsenseHandlerBase):
    def __init__(self, timeout=DEFAULT_TIMEOUT):
        super().__init__(timeout)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.is_running = False
        self.wait = True

    def start_camera(self):
        try:
            self.profile = self.pipeline.start(self.config)
            self.is_running = True
        except Exception as e:
            print(f"Failed to start camera: {e}")

    def stop_camera(self):
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False

    def __del__(self):
        if self.is_running:
            self.stop_camera()
    
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
            if wait:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
            else:  
                # Poll for frames
                frames = self.pipeline.poll_for_frames()
            
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
        
        except RuntimeError as e:
            # Specifically ignore null pointer exceptions silently
            # This is known and will spam during poll_for_frames() wait time
            if "null pointer passed for argument 'frame'" in str(e):
                return None, None
            else:
                raise  # Re-throw exception if it's not the handled type
        
    def get_3D_pose(self, depth_frame, shape):
        """
        Calculate the centroid of a given shape (either bbox or polygon) and retrieve the depth value at the centroid.

        Parameters:
        - depth_frame (rs.frame): Depth frame from the Intel Realsense camera.
        - shape (list/tuple): Can be a bbox as [x1, y1, x2, y2] or a polygon as [(x1, y1), (x2, y2), ...].

        Returns:
        - Tuple: Depth value at the centroid, or None if no depth value is available, along with the camera coordinates.
        """
        center_x, center_y = self._get_centroid_pixel(shape)
        # print(f'Center: {center_x}, {center_y}')
        if center_x is not None and center_y is not None:
            depth, x, y = self._get_depth_at_centroid(depth_frame, center_x, center_y)    
            return depth, x, y, center_x, center_y
        else:
            return 0, 0, 0, 0, 0

    def _get_centroid_pixel(self, shape):
        # Check if shape is a bbox or polygon
        if len(shape) == 4:
            # It's a bbox - make sure to close the loop
            points = np.array([
                [shape[0], shape[1]],  # Top-left corner
                [shape[2], shape[1]],  # Top-right corner
                [shape[2], shape[3]],  # Bottom-right corner
                [shape[0], shape[3]],  # Bottom-left corner
                [shape[0], shape[1]]   # Back to Top-left corner to close the loop
            ], dtype=np.int32)
        
        points = shape.reshape((-1, 1, 2))  # Reshape for OpenCV

        # Calculate the moments
        M = cv2.moments(points)
        # print(f"Moment calculation: {M}")
        if M["m00"] > 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return center_x, center_y
        else:
            print("Calculated zero area for the shape.")
            return None, None
        
    def _get_depth_at_centroid(self, depth_frame, center_x, center_y):
        """
        Get the depth value at the centroid based on the provided depth_frame and centroid coordinates.

        Parameters:
            - depth_frame (rs.frame): Depth frame from the Intel Realsense camera.
            - center_x (int): X-coordinate of the centroid.
            - center_y (int): Y-coordinate of the centroid.

        Returns:
            Tuple: Depth value at the centroid, or None if no depth value is available, along with the camera coordinates.
        """
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
                zxy_pose = z, x, y
        # Return the depth, x, and y coordinates
        return zxy_pose
        
    
    
