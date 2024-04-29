from Tracker_segmentation import Tracker
from pathlib import Path
import traceback
import torch

def main():
    base_path = Path(__file__).resolve().parent
    model_path = str(base_path / "weights" / "slide1-3-tune6.pt")
    # print(f"Model path: {model_path}")
    tracker = Tracker(model_path, log=True, display=True)

    try:
        tracker.run_model()
    except Exception as e:
        print("An unhandled exception occurred!")
        print(str(e))
        print(traceback.format_exc())  
    tracker.uninit()

if __name__ == "__main__":
    main()