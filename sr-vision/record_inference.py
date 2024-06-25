from TrackerSegmentation import TrackerSegmentation as Tracker
from InferenceRecorder import InferenceRecorder
from pathlib import Path
import traceback
import yaml
import torch
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    base_path = Path(__file__).resolve().parent
    config_path = str(base_path / 'model_configs' / 'yolo_config.yaml')
    output_path = str(base_path / 'record' / 'inference.avi')
    
    config = load_config(config_path)
    
    # Update the model_path with the base path
    config['model_path'] = str(base_path / config['model_path'])
    
    # Create tracker instance with config parameters
    tracker = Tracker(**config)

    recorder = InferenceRecorder(fps=15.0)

    try:
        tracker.start_camera()
        recorder.initialize_writer((640, 480), output_path)
        while True:
            tracker.update()
            display_frame = tracker.get_display_frame()
            recorder.write_frame(display_frame)
        # tracker.run_model()
    except Exception as e:
        print("An unhandled exception occurred!")
        print(str(e))
        print(traceback.format_exc())  
    tracker.uninit()
    recorder.release()

if __name__ == "__main__":
    main()











