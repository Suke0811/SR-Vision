from Tracker_detection import Tracker
from pathlib import Path

def main():
    base_path = Path(__file__).resolve().parent
    model_path = str(base_path / "weights" / "yolov8m.pt")
    # print(f"Model path: {model_path}")
    tracker = Tracker(model_path)

    tracker.run_model()

    tracker.uninit()

if __name__ == "__main__":
    main()