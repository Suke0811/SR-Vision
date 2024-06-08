# Using SR-Vision Yolov8 Inference:
Look at main.py for an example script on how to initialize and run the stack.
### Params:
``` yaml
model_path: absolute path to the weights as a string
classes: list of names for the classes in the model (default name will show as '???')
colors: dictionary of colors with the name of the class as a key
log: print out debugging and timing to the terminal
det_conf: confidence threshold for model detection
iou: threshold for intersection of union for bounding boxes
```
### update() function:
``` python3
tracker.update()
```
update() function runs the stack with the specified model, generating a 2D matrix with all the detected/segmented objects.

### get_positions() function:
``` python3
tracker.get_positions()
```
Returns 2D matrix of positions of detected/segmented objects:
```
[[class id, x, y, z]] 
```
xyz positions for camera frame while pointing forward: 
- position x is right
- positive y is down
- positive z is out