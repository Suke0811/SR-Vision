
DEFAULT_TIMEOUT = 100

class IntelRealsenseHandlerBase:
    def __init__(self, timeout=DEFAULT_TIMEOUT, *args, **kwargs):
        self.pipe = None
        self.timeout = timeout
        pass

    def init(self):
        """
        Initialize the device
        """
        raise NotImplementedError

    def start_camera(self, *args, **kwargs):
        raise NotImplementedError

    def get_frames(self, wait=False, *args, **kwargs):
        try:
            if wait:
                frames = self.pipe.wait_for_frames(self.timeout)
            else:
                frames = self.pipe.poll_for_frames()
        except RuntimeError:
            pass
        except KeyboardInterrupt:
            self.stop_camera()



    def stop_camera(self, *args, **kwargs):
        raise NotImplementedError


    def __del__(self):
        self.stop_camera()


class SegmentationHandlerBase:
    def __init__(self, *args, **kwargs):
        pass

    def segmentation(self, frame, *args, **kwargs):
        raise NotImplementedError


class FrameHandlerBase:
    def __init__(self, *args, **kwargs):
        pass

    def frame_to_xyz(self, frame, depth, *args, **kwargs):
        raise NotImplementedError

    def frame_to_distance(self, norm: int=2):
        raise NotImplementedError


