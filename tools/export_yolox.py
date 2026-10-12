#!/usr/bin/env python3
"""Rebuild yolox_models/yolox_tiny_int8_416x256.onnx from the official YOLOX-tiny weights.

Only needed to recreate or change the model; the app itself needs just onnxruntime. Needs PyTorch
and the YOLOX source (Apache 2.0):

  git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git /tmp/YOLOX
  curl -L -o /tmp/yolox_tiny.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth
  pip install torch onnx loguru tabulate
  PYTHONPATH=/tmp/YOLOX python tools/export_yolox.py --weights /tmp/yolox_tiny.pth \\
      --exp /tmp/YOLOX/exps/default/yolox_tiny.py --calib-dir <folder of camera .jpg frames>

416x256 matches a 16:9 camera at the same detail as a 416x416 input, without grey padding.
Quantisation: onnxruntime static int8, QDQ, per-channel, MinMax (as det3 did).
"""
import argparse
import glob
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "yolox_models"


def letterbox(img, h, w):
    r = min(h / img.shape[0], w / img.shape[1])
    canvas = np.full((h, w, 3), 114, np.uint8)
    nh, nw = int(img.shape[0] * r), int(img.shape[1] * r)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    return canvas.transpose(2, 0, 1)[None].astype(np.float32)  # BGR, 0-255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--exp", required=True, help="YOLOX/exps/default/yolox_tiny.py")
    ap.add_argument("--calib-dir", required=True, help="folder with .jpg frames from the target camera")
    ap.add_argument("--size", default="416x256", help="WxH, both multiples of 32")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from yolox.exp import get_exp
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    w, h = (int(v) for v in a.size.lower().split("x"))
    model = get_exp(a.exp, None).get_model()
    model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=False)["model"])
    model = replace_module(model, nn.SiLU, SiLU).eval()
    model.head.decode_in_inference = False  # grid/stride decode happens in yolox_backend.py

    calib = sorted(glob.glob(os.path.join(a.calib_dir, "*.jpg")))
    if not calib:
        raise SystemExit(f"no .jpg files in {a.calib_dir}")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.files = iter(calib)

        def get_next(self):
            f = next(self.files, None)
            return None if f is None else {"images": letterbox(cv2.imread(f), h, w)}

    OUT.mkdir(exist_ok=True)
    out = OUT / f"yolox_tiny_int8_{w}x{h}.onnx"
    with tempfile.TemporaryDirectory() as td:
        fp32, pre = os.path.join(td, "fp32.onnx"), os.path.join(td, "pre.onnx")
        torch.onnx.export(model, torch.randn(1, 3, h, w), fp32, input_names=["images"], output_names=["output"],
                          opset_version=18, dynamo=False, do_constant_folding=True)
        quant_pre_process(fp32, pre, skip_symbolic_shape=True)
        quantize_static(pre, str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), calibrated on {len(calib)} images")


if __name__ == "__main__":
    main()
