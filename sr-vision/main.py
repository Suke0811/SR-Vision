from TrackerSegmentation import TrackerSegmentation as Tracker
from pathlib import Path
import traceback

def main():
    base_path = Path(__file__).resolve().parent
    model_path = str(base_path / "weights" / "yolov8s-seg.pt")
    # print(f"Model path: {model_path}")
    tracker = Tracker(model_path, log=False, display=True)

    try:
        tracker.run_model()
    except Exception as e:
        print("An unhandled exception occurred!")
        print(str(e))
        print(traceback.format_exc())  
    tracker.uninit()

if __name__ == "__main__":
    main()