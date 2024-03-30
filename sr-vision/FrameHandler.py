from Base import FrameHandlerBase
from Base import IntelRealsenseHandler
from Base import SegmentationHandler
import cv2
import numpy as np


class FrameHandler(FrameHandlerBase):
    pass

    def __init__(self, camera, *args, **kwargs):
        self.cam = camera
        self.positions = np.empty((0, 4), dtype=np.float32)
        self.classes = ['Door Handle', 'Door Knob']
        self.center_xy = np.empty()
    
    def get_positions(self, depth_frame, polygons):
        """
        Get the positions of detected objects in a 2D matrix.

        Parameters:
            polygons (list): A list of tuples containing the class and polygon of each detected object.

        Returns:
            numpy.ndarray: A 2D matrix containing the positions of the detected objects. Each row represents a detected object and contains the 
            class, x-coordinate, y-coordinate, and depth of the object.
        """
        # initialize 2D Matrix for positions of detected objects
        self.positions = np.empty((0, 4), dtype=np.float32)
        
        for cls, polygon in polygons:
            # Extract depth at centroid
            depth, x, y, center_x, center_y = self.cam.get_3D_pose(depth_frame, polygon)
            position = np.array([[cls, x, y, depth]])
            center = np.array([[center_x, center_y]])
            
            # Append center of detection to 2D Matrix
            self.center_xy = np.vstack((self.center_xy, center))
            # Append position of detection to 2D Matrix
            self.positions = np.vstack((self.positions, position))
            
        return self.positions
    
    def display_data(self, frame, polygons, bboxes):
        """
        A function to display data including bounding box, segmentation polygon, and label on the frame.

        Parameters:
        - box: The bounding box object
        - polygon: The segmentation polygon points
        - bbox: The bounding box coordinates
        - ID: The class ID
        - center_x: The x-coordinate of the centroid
        - center_y: The y-coordinate of the centroid

        Returns:
        This function applies the changes to the frame and does not return anything.
        """
        for position, center, polygon, box in zip(self.positions, self.center_xy, polygons, bboxes):
            # get 3D coordinates
            center_3d = position[1:4]
            # get centroid center pixel
            center_x, center_y = center
            # unpack box
            ID, confidence, bbox = box
            
            # get class name
            current_class_name = self.classes[ID]
            
            # Draw bounding box 
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            xyz_label = f"X: {center_3d[0]:.2f}, Y: {center_3d[1]:.2f}, Z: {center_3d[2]:.2f}"
            label = f"{current_class_name} {confidence:.2f} ({xyz_label})"
            
            # Draw segmentation polygon
            cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)
            
            # Draw a circle at the 2D centroid
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Adjust the y position of the label to sit on top or inside the bounding box
            # label_y_position = max(y1, label_height + baseline) 
            label_y_position = y1 - label_height + baseline + 4
            box_y_position = y1 - label_height - baseline 
            if (box_y_position < 0): 
                label_y_position = y1 + label_height - baseline + 6 
                box_y_position = y1 + label_height + baseline                      

            # Create a rectangle for the label background and put the text directly on the bounding box
            cv2.rectangle(frame, (x1, box_y_position), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame
        
        