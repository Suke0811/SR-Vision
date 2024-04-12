from Tracker_detection import Tracker
import pathlib as Path

def main(self):
    base_path = Path(__file__).parent
    model_path = str(base_path / "weights" / "yolov8m.pt")
    tracker = Tracker(model_path)

    tracker.run()

    tracker.uninit()

if __name__ == "__main__":
    main()