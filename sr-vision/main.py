from TrackerSegmentation import TrackerSegmentation as Tracker
from pathlib import Path
import traceback
import torch

def main():
    base_path = Path(__file__).resolve().parent
    model_path = str(base_path / "weights" / "yolov8s-seg.pt")
    # print(f"Model path: {model_path}")
    classes = [] # list for class labels e.g.('shiba')
    colors = {} # dict for class label colors e.g.{'shiba': (255, 0, 0)}
    tracker = Tracker(model_path, classes=classes, colors=colors, log=False, display=True)

    try:
        tracker.run_model()
    except Exception as e:
        print("An unhandled exception occurred!")
        print(str(e))
        print(traceback.format_exc())  
    tracker.uninit()

if __name__ == "__main__":
    main()