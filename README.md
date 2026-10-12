# Pi 5 Person Detection

Real-time **person detection** on a Raspberry Pi 5 CPU with **YOLOX-tiny** (int8 ONNX, onnxruntime).
About 20 FPS live with the Pi camera, no accelerator needed. Everything here is free to use commercially:
YOLOX and its weights are Apache 2.0, onnxruntime is MIT, OpenCV is Apache 2.0, Flask is BSD.

Overview with measured results: **https://pratyush150.github.io/pi5-yolo11-detection/** ([PDF](docs/pi5-person-detector.pdf)).

## Use it in an application

```python
import cv2
from person_detector import PersonDetector

detector = PersonDetector()              # loads the model once
frame = cv2.imread("photo.jpg")          # any BGR image (OpenCV, or Picamera2 "RGB888")
for p in detector.detect(frame, video=False):   # video=False for single photos
    print(p["conf"], p["box"])           # box = [x1, y1, x2, y2] in frame pixels
```

For a video stream keep one `PersonDetector` per camera and call `detect(frame)` on every frame. It then
also ignores person-like things that never move (a chair back, a coat) and reports a person only after
they appeared in 3 of the last 5 frames. Options: `PersonDetector(conf=0.45, motion=True, confirm=True, threads=3)`.

To read the camera too, `camera.LatestFrame("auto", rotate=180)` gives the newest frame from the Pi camera
(or USB webcam 0, or a video file) with `read()`.

## What `detect()` does

| Step | Detail |
|---|---|
| Model | `yolox_models/yolox_tiny_int8_416x256.onnx`: official YOLOX-tiny COCO weights, exported at 416x256 to match a 16:9 camera without grey padding, int8 quantised |
| Person only | only the COCO "person" score is used |
| Threshold `conf=0.45` | still catches partly hidden people; on test footage a person half behind a box was found in every frame, with no false person boxes |
| Motion check (video) | a person box is accepted only once something moved inside it, then followed even if they keep still. On 250 COCO photos of furniture without people the model alone saw a "person" in 4 (a chair back, a wig, clothes, a figurine); repeated as a still scene, all were rejected. A nearly still person on the Pi camera was accepted within ~1.5 s |
| Confirm (video) | a person must appear in 3 of the last 5 frames (~0.5 s at 20 FPS) |

Speed on a Raspberry Pi 5 (1280x720 frames): 29 ms per frame for the model, ~20 FPS live with camera and web UI.

## Web UI (for testing)

```bash
.venv/bin/python app.py --rotate 180          # Pi camera mounted upside down; http://<pi-ip>:8000
.venv/bin/python app.py --source 0 --port 8090
.venv/bin/python app.py --source clip.mp4
```

Shows the live feed with boxes, FPS, the people in view, a change log, a person-threshold slider, switches
for the motion check and 3-frame confirmation, pause and snapshot. Photos (`--source x.jpg`) skip the video checks. If no camera is connected it says so and keeps retrying.

## Setup

```bash
sudo apt install -y python3-picamera2         # only for the Pi camera module
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `person_detector.py` | the API for applications |
| `yolox_backend.py` | model loading, pre/post-processing |
| `filters.py` | motion check and 3-of-5-frame confirmation |
| `camera.py` | Pi camera / webcam / video reader that always returns the newest frame |
| `app.py`, `ui/` | web UI for testing |
| `tools/export_yolox.py` | rebuilds the model from the official weights (needs PyTorch, not needed at runtime) |

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q
```

They download three COCO photos once (a skier, two baseball players with one partly hidden, two cats)
and run on GitHub Actions for every push.

## Licences

YOLOX and the YOLOX-tiny weights: Apache 2.0 (Megvii). Keep this notice and the Apache licence when you
ship the model. COCO test photos: CC BY 4.0, downloaded at test time, not included.
