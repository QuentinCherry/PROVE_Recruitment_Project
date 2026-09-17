# PROVE Recruitment Project

Live road object detection using a YOLO model on a live camera feed. Combined with a road line locator and 3D visualizer of detected objects.

## Getting Started
Get started by running the following in terminal to install required packages.

```bash
pip install -r requirements.txt
```

## Main File

[Detect.py](Detect.py) is the main file in this project. 

it can be ran with
```bash
python Detect.py
```

Pressing 'q' on any of the windows will quit the program


## Switching Model

This repo has several basic models, with more downloadable via the Ultralytics website
The selected YOLO model is loaded on [Detect.py:12](Detect.py)

```python
model = YOLO("exp.pt") # Model to use
```

Other models included in the repo are:

- `exp.pt`
- `dailyroaddetection.pt`
- `yolov5s.pt`
- `yolov5su.pt`

Note exp.pt is the only yolov11 model pre-packaged

Core ML variants of some models (`.mlpackage`) are also included (`exp.mlpackage`,
`dailyroaddetection.mlpackage`, `yolov5su.mlpackage`) for use on Apple platforms

Using these models will instead use the ARM chip's NPU instead of GPU

Example:

```python
model = YOLO("exp.mlpackage") # Model to use
```

## Other settings

Detect.py has a few other variables, while they aren't required to change they are optional.
- Line 11: `cam = cv2.VideoCapture(0)` - Camera to use, default is usually `0` but others may be used if installed
- Line 13: `model.overrides["imgsz"] = 1280` - Resolution to use, higher value = better accuracy/range but significantly higher compute cost. **Must be a multiple of 32.**
- Line 16-19: `horizontal_fov`, `vertical_fov`, `horizontal_pixels`, `vertical_pixels` - Camera values used to convert bounding boxes into horizontal/vertical angles 
*Not required for operation, but is used for angle calculations*
- Line 23: `show_render = False` - Enables and disables the VPython 3D scene rendering. Do note its somewhat buggy and may freeze up.


## Other programs

`bulkdownloader.py` - The webcrawler used for gathering training data
`bulkRenamer.py` - renames files in mass to avoid filename conflicts with training, do note it clones all files.
`checkcuda.py` - simply checks if the hardware has access to cuda cores
`modelExport.py` - exports a `.pt` model to CoreML `.mlpackage` for use with Apple neural engine