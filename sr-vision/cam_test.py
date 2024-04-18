from Tracker_detection import Tracker
from IntelRealsenseHandler import IntelRealsenseHandler
from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    camera = IntelRealsenseHandler()
    camera.start_camera()
    model = YOLO('/home/romela/Alvin-files/SR-Vision/sr-vision/weights/yolov8m.pt')
    while True:
        depth_frame, color_frame  = camera.get_frames(True)
        results = model(color_frame)
        # print('showing frames')
        cv2.imshow('camera test', color_frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()