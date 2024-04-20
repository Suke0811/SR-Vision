from Tracker_segmentation import Tracker
from pathlib import Path

def main():
    base_path = Path(__file__).resolve().parent
    model_path = str(base_path / "weights" / "yolov8m-seg.pt")
    # print(f"Model path: {model_path}")
    tracker = Tracker(model_path, log=False, display=True)

    tracker.run_model()

    tracker.uninit()

if __name__ == "__main__":
    main()