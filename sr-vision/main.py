from TrackerSegmentation import TrackerSegmentation as Tracker
from pathlib import Path
import traceback
import torch

def main():
    base_path = Path(__file__).resolve().parent
    # model_path = str(base_path / "weights" / "slide2-1-tune20.onnx")
    model_path = str(base_path / "weights" / "slideS3-1-tune8.pt")
    # print(f"Model path: {model_path}")
    classes = ['handle', 'stair']
    colors = {'handle': (255, 0, 0), 'stair': (0, 255, 0)}
    tracker = Tracker(model_path, classes=classes, colors=colors, log=True, display=True)

    try:
        tracker.run_model()
    except Exception as e:
        print("An unhandled exception occurred!")
        print(str(e))
        print(traceback.format_exc())  
    tracker.uninit()

if __name__ == "__main__":
    main()