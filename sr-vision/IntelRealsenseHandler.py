from .Base import DEFAULT_TIMEOUT
from Base import IntelRealsenseHandlerBase
import pyrealsense2 as rs
import numpy as np
import cv2


class IntelRealsenseHandler(IntelRealsenseHandlerBase):
    pass

    def __init__(self, timeout=DEFAULT_TIMEOUT):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.wait = True # use poll for frames or wait for frames
        
        # Align the two cameras since there is physical offset
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        
        # Variable to hold camera intrinsics
        self.intrinsics = None
        
        # np arrays for both depth and color images
        self.depth_frame = None
        self.color_frame = None
        
    def start_camera(self):
        # Start the pipeline
        self.profile = self.pipeline.start(self.config)
        # Get the camera intrinsics from the color stream
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    
    def stop_camera(self):
        # Stop the pipeline
        self.pipeline.stop()
    
    '''Getters:'''
    
    def get_color_frame(self):
        return self.color_frame
    
    def get_depth_frame(self):
        return self.depth_frame
    
    def get_intrinsics(self):
        return self.intrinsics
    
    def get_frames(self):
        try:
            # Attempt to retrieve the next set of frames
            if self.wait:
                frames = self.pipeline.wait_for_frames()
            else:
                frames = self.pipeline.poll_for_frames()
                
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
            
    def get_3D_pose(self, depth_frame, polygon):
        """
        Calculate the centroid of the given polygon and retrieve the depth value at the centroid. 

        Parameters:
        polygons (list): List of points representing the polygon.

        Returns:
        float or None: Depth value at the centroid, or None if no depth value is available.
        float: X coordinate of the centroid in camera coordinates.
        float: Y coordinate of the centroid in camera coordinates.
        int: X pixel coordinate of the centroid in the depth frame.
        int: Y pixel coordinate of the centroid in the depth frame.
        """
        # Calculate the centroid
        M = cv2.moments(np.array(polygon, dtype=np.int32))
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            return None, 0, 0

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
                return z, x, -y, center_x, center_y
            else:
                return None, 0, 0, 0, 0
        else:
            return None, 0, 0, 0, 0
        
    '''Setters:'''
    
