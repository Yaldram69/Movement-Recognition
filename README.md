# Movement Recognition

A webcam-based moving object tracker. It runs YOLOv8 detection on each frame, tracks objects across frames with DeepSORT, and draws bounding boxes only for objects whose centre has moved more than a fixed threshold between frames (new tracks are also shown on first appearance).

## Features

- YOLOv8 object detection (Ultralytics) on live webcam frames
- Multi-object tracking with DeepSORT (`deep_sort_realtime`)
- Per-track centre position history; objects are drawn only when they move more than 10 pixels
- Live display window; press `q` to quit

## Project structure

- `Run.py` - main script: webcam capture, YOLOv8 detection, DeepSORT tracking, movement filter, display
- `README.txt` - pip install commands and run note from the original project
- `.idea/` - PyCharm project metadata (IDE settings, not needed to run the code)

## Requirements

Packages imported by `Run.py`:

- opencv-python (`cv2`)
- torch
- numpy
- ultralytics (`YOLO`)
- deep_sort_realtime

Install:

```bash
pip install opencv-python ultralytics torch numpy deep_sort_realtime
```

## Installation

```bash
pip install opencv-python ultralytics torch numpy deep_sort_realtime
```

A CUDA-enabled PyTorch build is optional; the script runs on CPU.

## Usage

Run with a webcam connected (device index 0):

```bash
python Run.py
```

- Input: webcam (`cv2.VideoCapture(0)`)
- Output: window titled "Moving Object Tracker"
- Press `q` to quit

## Notes

- The script loads `yolov8m.pt`. This weights file is not included in this folder; Ultralytics will download it automatically on first run if it is available. The comment in the script suggests `yolov8n.pt` for speed or `yolov8s.pt` for accuracy.
- `import torch` is present but `torch` is not used directly in the script.
- The movement threshold (`MOVEMENT_THRESHOLD = 10`) is in pixels and is not normalised for camera resolution.
- Tracks are kept in a dictionary that is never pruned, so memory grows with the number of unique track IDs over a long session.
