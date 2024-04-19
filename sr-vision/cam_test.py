from Tracker_detection import Tracker
from IntelRealsenseHandler import IntelRealsenseHandler
from DetectionHandler import DetectionHander
from FrameHandler import FrameHandler
from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    model_path = '/home/romela/Alvin-files/SR-Vision/sr-vision/weights/yolov8m.pt'
    camera = IntelRealsenseHandler()
    detector = DetectionHander(model_path, display=True)
    frame_handler = FrameHandler(camera)
    camera.start_camera()
    
    while True:
        depth_frame, color_frame = camera.get_frames(True)
        bboxes = detector.detection(color_frame)
        positions = frame_handler.get_xyz(depth_frame, bboxes)
        annotated_frame = frame_handler.display_data(color_frame, bboxes)
        # print('showing frames')
        cv2.imshow('camera test', annotated_frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()