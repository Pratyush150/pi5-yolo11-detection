"""YOLOX-tiny person detector on the CPU (onnxruntime).

Model: yolox_models/yolox_tiny_int8_416x256.onnx, the official YOLOX-tiny COCO weights (Apache 2.0)
exported at 416x256 to match the 16:9 camera and quantized to int8 (see tools/export_yolox.py).
Processing follows YOLOX: top-left letterbox on a 114-grey canvas, raw 0-255 BGR input, grid/stride
decode of the [1, N, 85] head, score = objectness x person class, NMS at 0.45.
"""
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "yolox_models" / "yolox_tiny_int8_416x256.onnx"
PERSON = 0  # COCO class index


class YoloxBackend:
    """The raw model: load(), then infer(frame, conf) -> [{"label": "person", "conf", "box": [x1, y1, x2, y2]}]."""

    def __init__(self, model_path=MODEL_PATH, threads=3, nms_iou=0.45):
        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found")
        self.threads, self.nms_iou = threads, nms_iou  # 3 threads leave a core for the camera and web server
        self.label = "YOLOX-tiny · person"
        self.input_size = None  # (w, h), read from the model in load()

    def load(self):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.iname, (self.in_h, self.in_w) = inp.name, inp.shape[2:4]
        self.input_size = (self.in_w, self.in_h)
        grids, strides = [], []
        for s in (8, 16, 32):
            gh, gw = self.in_h // s, self.in_w // s
            yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((gh * gw, 1), s))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def infer(self, frame, conf):
        h, w = frame.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        canvas[:int(h * r), :int(w * r)] = cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)  # BGR, 0-255
        p = self.sess.run(None, {self.iname: x})[0][0]

        scores = p[:, 4] * p[:, 5 + PERSON]
        keep = scores >= conf
        if not keep.any():
            return []
        xy = (p[keep, :2] + self.grid[keep]) * self.stride[keep]
        wh = np.exp(p[keep, 2:4]) * self.stride[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        scores = scores[keep]
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        idx = np.array(cv2.dnn.NMSBoxes(xywh, scores.tolist(), conf, self.nms_iou)).flatten()
        return [{"label": "person", "conf": float(scores[i]), "box": boxes[i].tolist()} for i in idx]
