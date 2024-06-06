from Base import FrameHandlerBase, IntelRealsenseHandlerBase, SegmentationHandlerBase
from IntelRealsenseHandler import IntelRealsenseHandler
from DetectionHandler import DetectionHander
import cv2
import numpy as np
import traceback


class FrameHandler(FrameHandlerBase):
    def __init__(self, camera, classes=[], colors={},*args, **kwargs):
        self.cam = camera
        self.positions = np.empty((0, 4), dtype=np.float32)
        self.center_xy = np.empty((0, 2), dtype=np.float32)  
        self._classes = classes
        self._colors = colors
    
    def get_xyz(self, depth_frame, polygons, *args, **kwargs):
        return self._get_positions(depth_frame, polygons)
    
    def get_depth(self, norm: int=2):
        pass
    
    def _get_positions(self, depth_frame, polygons, *args, **kwargs):
        """
        Get the positions of detected objects in a 2D matrix.

        Parameters:
            polygons (list): A list of tuples containing the class and polygon of each detected object.
            - polygons can also be a bbox

        Returns:
            numpy.ndarray: A 2D matrix containing the positions of the detected objects. Each row represents a detected object and contains the 
            class, x-coordinate, y-coordinate, and depth of the object.
        """
        # initialize 2D Matrix for positions of detected objects / refresh for each cycle
        self.positions = np.empty((0, 4), dtype=np.float32)
        self.center_xy = np.empty((0, 2), dtype=np.float32)
        
        for cls_, confidence, polygon in polygons:
            # Extract depth at centroid
            depth, x, y, center_x, center_y = self.cam.get_3D_pose(depth_frame, polygon)
            # print(f'Depth: {depth} X: {x} Y: {y}')
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
        - polygons: The segmentation polygon points or None
        - bboxes: The bounding box objects containing class ID, confidence, and coordinates
        Returns:
        - The frame with the applied changes
        """
        # Ensure polygons is iterable and of the same length as bboxes
        # Modular to be used with both detectiona and segmentation  
        if polygons is None:
            polygons = [None] * len(bboxes)

        if bboxes is not None:
            try:
                for position, center, box, polygon in zip(self.positions, self.center_xy, bboxes, polygons):
                    center_3d = self._get_3d_coordinates(position)
                    center_x, center_y = self._get_centroid_pixel(center)
                    id_, confidence, bbox = self._unpack_shape(box)
                    current_class_name = self._get_class_name(id_)
                    self._draw_bounding_box(frame, current_class_name, bbox)
                    label = self._create_label(current_class_name, confidence, center_3d)

                    # Only draw the polygon if it is not None
                    if polygon is not None:
                        _, _, polygon = self._unpack_shape(polygon)
                        self._draw_segmentation_polygon(frame, current_class_name, polygon)

                    self._draw_centroid(frame, current_class_name, center_x, center_y)
                    self._draw_label(frame, current_class_name, bbox, label)

            except Exception as e:
                # print(e)
                # print(traceback.format_exc())
                pass
        return frame


    def _get_3d_coordinates(self, position):
        return position[1:4]

    def _get_centroid_pixel(self, center):
        center_x, center_y = int(center[0]), int(center[1])
        return center_x, center_y

    def _unpack_shape(self, shape):
        id_, confidence, _shape = shape
        return id_, confidence, _shape

    def _get_class_name(self, id_):
        try:
            return self._classes[id_]
        except IndexError:
            return "???"


    def _draw_bounding_box(self, frame, class_name, bbox):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self._colors[class_name], 2)

    def _create_label(self, class_name, confidence, center_3d):
        xyz_label = f"X: {center_3d[0]:.2f}, Y: {center_3d[1]:.2f}, Z: {center_3d[2]:.2f}"
        return f"{class_name} {confidence:.2f} ({xyz_label})"

    def _draw_segmentation_polygon(self, frame, class_name, polygon):
        # cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], isClosed=True, color=self._colors[class_name], thickness=2)
        cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)

    def _draw_centroid(self, frame, class_name, center_x, center_y):
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    def _draw_label(self, frame, class_name, bbox, label):
        x1, y1, _, _ = bbox
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        label_y_position = y1 - label_height + baseline + 4
        box_y_position = y1 - label_height - baseline
        if box_y_position < 0:
            label_y_position = y1 + label_height - baseline + 6
            box_y_position = y1 + label_height + baseline
        
        cv2.rectangle(frame, (x1, box_y_position), (x1 + label_width, y1), self._colors[class_name], cv2.FILLED)
        cv2.putText(frame, label, (x1, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        