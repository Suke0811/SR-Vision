from Base import FrameHandlerBase, IntelRealsenseHandler, SegmentationHandler
from Tracker_detection import Tracker_detection
from Tracker_segmentation import Tracker_segmentation
import cv2
import numpy as np


class FrameHandler(FrameHandlerBase):
    def __init__(self, camera, tracker_type='detection', *args, **kwargs):
        self.cam = camera
        self.positions = np.empty((0, 4), dtype=np.float32)
        self.tracker = None
        self.tracker_type = tracker_type
        self.center_xy = np.empty((0, 2), dtype=np.float32)  # Corrected initialization

        if tracker_type == 'detection':
            self.tracker = Tracker_detection()
        elif tracker_type == 'segmentation':
            self.tracker = Tracker_segmentation()
        else:
            raise ValueError("Invalid tracker type specified.")
    
    @property
    def classes(self):
        # Assuming both Tracker_detection and Tracker_segmentation have a 'classes' attribute
        return self.tracker.classes if self.tracker else None

    @classes.setter
    def classes(self, classes):
        if self.tracker:
            self.tracker.classes = classes
    
    def get_xyz(self, depth_frame, polygons, *args, **kwargs):
        return self._get_positions(depth_frame, polygons)
    
    def get_depth(self, norm: int=2):
        pass
    
    def _get_positions(self, depth_frame, polygons, *args, **kwargs):
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
        
        for cls_, polygon in polygons:
            # Extract depth at centroid
            depth, x, y, center_x, center_y = self.cam.get_3D_pose(depth_frame, polygon)
            position = np.array([[cls_, x, y, depth]])
            center = np.array([[center_x, center_y]])
            
            # Append center of detection to 2D Matrix
            self.center_xy = np.vstack((self.center_xy, center))
            # Append position of detection to 2D Matrix
            self.positions = np.vstack((self.positions, position))
            
        return self.positions
    
    '''DISPLAY FUNCTIONS'''

    def display_data(self, frame, bboxes, polygons=None):
        """
        A function to display data including bounding box, segmentation polygon, and label on the frame.
        Parameters:
        - frame: The input frame
        - polygons: The segmentation polygon points
        - bboxes: The bounding box objects containing class ID, confidence, and coordinates
        Returns:
        - The frame with the applied changes
        """
        for position, center, box, polygon in zip(self.positions, self.center_xy, bboxes, polygons):
            center_3d = self._get_3d_coordinates(position)
            center_x, center_y = self._get_centroid_pixel(center)
            id_, confidence, bbox = self._unpack_box(box)
            current_class_name = self._get_class_name(id_)
            self._draw_bounding_box(frame, bbox)
            label = self._create_label(current_class_name, confidence, center_3d)
            # check if drawing for segmentation or detection
            if polygon is not None:
                self._draw_segmentation_polygon(frame, polygon)
            self._draw_centroid(frame, center_x, center_y)
            self._draw_label(frame, bbox, label)
        
        return frame

    def _get_3d_coordinates(self, position):
        return position[1:4]

    def _get_centroid_pixel(self, center):
        center_x, center_y = center
        return center_x, center_y

    def _unpack_box(self, box):
        id_, confidence, bbox = box
        return id_, confidence, bbox

    def _get_class_name(self, id_):
        try:
            return self._classes[id_]
        except IndexError:
            return "???"


    def _draw_bounding_box(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def _create_label(self, class_name, confidence, center_3d):
        xyz_label = f"X: {center_3d[0]:.2f}, Y: {center_3d[1]:.2f}, Z: {center_3d[2]:.2f}"
        return f"{class_name} {confidence:.2f} ({xyz_label})"

    def _draw_segmentation_polygon(self, frame, polygon):
        cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)

    def _draw_centroid(self, frame, center_x, center_y):
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    def _draw_label(self, frame, bbox, label):
        x1, y1, _, _ = bbox
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        label_y_position = y1 - label_height + baseline + 4
        box_y_position = y1 - label_height - baseline
        if box_y_position < 0:
            label_y_position = y1 + label_height - baseline + 6
            box_y_position = y1 + label_height + baseline
        
        cv2.rectangle(frame, (x1, box_y_position), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (x1, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        