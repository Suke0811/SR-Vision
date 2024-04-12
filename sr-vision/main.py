from Tracker_detection import Tracker
from pathlib import Path

def main():
    base_path = Path(__file__).resolve().parent
    model_path = base_path / "weights" / "yolov8m.pt"
    tracker = Tracker(model_path)

    tracker.run_model()

    tracker.uninit()

if __name__ == "__main__":
    main()