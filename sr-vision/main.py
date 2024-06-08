from TrackerSegmentation import TrackerSegmentation as Tracker
from pathlib import Path
import traceback
import yaml
import torch

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    base_path = Path(__file__).resolve().parent
    config_path = str(base_path / 'model_configs' / 'yolo_config.yaml')
    
    config = load_config(config_path)
    
    # Update the model_path with the base path
    config['model_path'] = str(base_path / config['model_path'])
    
    # Create tracker instance with config parameters
    tracker = Tracker(**config)

    try:
        tracker.start_camera()
        while True:
            tracker.update()
            print(tracker.get_positions())
        # tracker.run_model()
    except Exception as e:
        print("An unhandled exception occurred!")
        print(str(e))
        print(traceback.format_exc())  
    tracker.uninit()

if __name__ == "__main__":
    main()
