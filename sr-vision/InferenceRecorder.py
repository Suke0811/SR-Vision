import cv2
from pathlib import Path

class InferenceRecorder:
    def __init__(self, fps=30.0, codec='XVID'):
        """
        Initialize the FrameRecorder with output path, frame rate, and codec.
        Args:
        - output_path (str): Path to save the output video.
        - fps (float): Frames per second for the output video.
        - codec (str): Codec to use for encoding the video.
        """
        self.fps = fps
        self.codec = codec
        self.is_initialized = False

    def initialize_writer(self, frame_shape, output_path):
        """
        Initialize the video writer object based on the first frame received.
        Args:
        - frame_shape (tuple): The shape of the frame to be recorded.
        """
        width, height = frame_shape
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        if not self.out.isOpened():
            print("Error: Failed to initialize video writer")
            return
        self.is_initialized = True

    def write_frame(self, frame):
        """
        Write a frame to the video file. Initializes video writer on the first call.
        Args:
        - frame (numpy.ndarray): The frame to write to the video.
        """
        self.out.write(frame)
        # print("INFERENCE FRAME WRITTEN")


    def release(self):
        """
        Release the video writer object.
        """
        if self.is_initialized:
            self.out.release()

    def __del__(self):
        self.release()
