"""Person detection web UI (YOLOX-tiny on the CPU). Open http://<pi-ip>:8000 from any device on the network.

Usage:
  python app.py                         # auto: Pi camera, else USB webcam 0
  python app.py --rotate 180            # camera mounted upside down
  python app.py --source 0              # USB webcam
  python app.py --source clip.mp4       # video file (loops)
  python app.py --source photo.jpg      # still image (repeats)
  python app.py --port 8090             # another port

Applications that only need the detections should use person_detector.PersonDetector directly.
"""
import argparse
import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

from camera import IMAGE_EXTS, LatestFrame
from person_detector import PersonDetector

HERE = Path(__file__).resolve().parent
BOX_COLOUR = (40, 170, 255)  # BGR amber

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # no spam while retrying a missing camera


def draw(frame, people):
    h, w = frame.shape[:2]
    thick = max(2, round(min(h, w) / 360))
    scale = max(0.5, min(h, w) / 900)
    for p in people:
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOUR, thick)
        text = f"person {p['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 if y1 - th - 8 >= 0 else y1 + th + 8
        cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 6, ty), BOX_COLOUR, -1)
        cv2.putText(frame, text, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


class Detector:
    """Runs capture + detection in one background thread and keeps the latest result for the UI."""

    def __init__(self, source, rotate=0, person_conf=0.45, confirm=True, motion=True, threads=3):
        self.source, self.rotate, self.threads = source, rotate, threads
        self.settings = {"person_conf": person_conf, "motion": motion, "confirm": confirm, "paused": False}
        self.people = None  # PersonDetector, created in the worker thread
        self.label = "YOLOX-tiny · person"
        self.lock = threading.Lock()
        self.jpeg = None
        self.raw = None
        self._latest = None  # (frame, people); drawn + encoded only when a viewer asks
        self.state = {"status": "starting", "message": "Loading model...", "fps": 0.0,
                      "infer_ms": 0.0, "objects": [], "counts": {}, "frame_size": None}
        self.events = deque(maxlen=50)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the worker before the interpreter exits so no thread is left mid-inference."""
        self._stop.set()
        self._thread.join(timeout=10)

    def update_settings(self, **changes):
        new = dict(self.settings, **changes)
        if self.people is not None and (new["confirm"], new["motion"]) != (self.settings["confirm"], self.settings["motion"]):
            self.people.reset()
        self.settings = new
        return new

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def _open(self):
        if Path(self.source).suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(self.source)
            if img is None:
                raise RuntimeError(f"Could not read image {self.source}")
            return None, cv2.rotate(img, cv2.ROTATE_180) if self.rotate == 180 else img
        try:
            return LatestFrame(self.source, self.rotate), None
        except SystemExit as e:
            raise RuntimeError(str(e)) from None

    def _run(self):
        try:
            self.people = PersonDetector(threads=self.threads)
            self.label = self.people.label
        except Exception as e:
            self._set(status="error", message=f"Could not load model: {e}")
            return
        last_count, fps, t_prev = None, 0.0, time.time()
        while not self._stop.is_set():
            try:
                stream, still = self._open()
            except Exception as e:
                self._set(status="no_camera", message=f"{e}. Retrying every 3 s.", fps=0.0)
                self._stop.wait(3)
                continue
            self._set(status="starting", message="Waiting for the first frame...")
            try:
                while not self._stop.is_set():
                    if still is not None:
                        ok, frame = True, still.copy()
                        self._stop.wait(0.05)
                    else:
                        ok, frame = stream.read()
                    if not ok or frame is None:
                        break
                    s = dict(self.settings)
                    if s["paused"]:
                        self._stop.wait(0.1)
                        continue
                    t0 = time.time()
                    found = self.people.detect(frame, conf=s["person_conf"], video=still is None,  # photos: no video checks
                                               motion=s["motion"], confirm=s["confirm"])
                    infer_ms = (time.time() - t0) * 1000
                    objects = [{"label": "person", "conf": round(p["conf"], 3)} for p in found]
                    if len(found) != last_count:
                        text = f"person x{len(found)}" if found else "nothing"
                        self.events.appendleft({"time": time.strftime("%H:%M:%S"), "text": text})
                        print(f"[{time.strftime('%H:%M:%S')}] Detected: {text}", flush=True)
                        last_count = len(found)
                    now = time.time()
                    fps = 0.85 * fps + 0.15 * (1 / max(now - t_prev, 1e-6))
                    t_prev = now
                    with self.lock:
                        self._latest, self.jpeg, self.raw = (frame, found), None, frame
                        self.state.update(status="live", message=f"Source: {self.source}", fps=round(fps, 1),
                                          infer_ms=round(infer_ms), objects=objects,
                                          counts={"person": len(found)} if found else {},
                                          frame_size=f"{frame.shape[1]}x{frame.shape[0]}")
            except Exception as e:
                self._set(status="error", message=f"Stream error: {e}")
            finally:
                if stream is not None:
                    stream.release()
            if self._stop.is_set():
                break
            if stream is not None and stream.is_file:
                continue  # loop video files
            self._set(status="no_camera", message="Camera disconnected. Retrying every 3 s.", fps=0.0)
            self._stop.wait(3)

    def snapshot(self, encode=True):
        """(annotated JPEG or None, state). Drawing and encoding happen here, outside the
        detection loop, so the model is fed frames back to back."""
        with self.lock:
            latest, jpeg = self._latest, self.jpeg
            state = dict(self.state, settings=dict(self.settings), events=list(self.events))
        if encode and jpeg is None and latest is not None:
            frame, found = latest
            jpeg = cv2.imencode(".jpg", draw(frame.copy(), found), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            with self.lock:
                if self._latest is latest:
                    self.jpeg = jpeg
        return jpeg, state


def create_app(det: Detector) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(HERE / "ui", "index.html")

    @app.get("/api/state")
    def state():
        _, s = det.snapshot(encode=False)
        s["model"] = det.label
        return jsonify(s)

    @app.post("/api/settings")
    def settings():
        body = request.get_json(force=True, silent=True) or {}
        changes = {}
        if "person_conf" in body:
            changes["person_conf"] = min(max(float(body["person_conf"]), 0.05), 0.95)
        if "confirm" in body:
            changes["confirm"] = bool(body["confirm"])
        if "motion" in body:
            changes["motion"] = bool(body["motion"])
        if "paused" in body:
            changes["paused"] = bool(body["paused"])
        return jsonify(det.update_settings(**changes))

    @app.get("/stream")
    def stream():
        def frames():
            last = None
            while True:
                jpeg, _ = det.snapshot()
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.03)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/raw.jpg")
    def raw():
        """Latest camera frame without boxes, for collecting test images."""
        with det.lock:
            frame = det.raw
        if frame is None:
            return "No frame yet", 503
        return Response(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(),
                        mimetype="image/jpeg")

    @app.get("/api/snapshot.jpg")
    def snapshot():
        jpeg, _ = det.snapshot()
        if jpeg is None:
            return "No frame yet", 503
        name = time.strftime("detection_%Y%m%d_%H%M%S.jpg")
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Content-Disposition": f"attachment; filename={name}"})

    return app


def main():
    ap = argparse.ArgumentParser(description="Person detection web UI (YOLOX-tiny on the CPU)")
    ap.add_argument("--source", default="auto", help="'auto', webcam index, 'picam', video or image path")
    ap.add_argument("--person-conf", type=float, default=0.45,
                    help="person confidence threshold (0.45 still catches partly hidden people)")
    ap.add_argument("--no-confirm", action="store_true", help="show detections immediately, without 3-of-5-frame confirmation")
    ap.add_argument("--no-motion", action="store_true", help="also accept person boxes where nothing has moved")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 180], help="180 for an upside-down camera")
    ap.add_argument("--threads", type=int, default=3, help="CPU threads for the model")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    det = Detector(args.source, rotate=args.rotate, person_conf=args.person_conf,
                   confirm=not args.no_confirm, motion=not args.no_motion, threads=args.threads)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # hide per-request lines
    print(f"Person detection UI on http://{args.host}:{args.port}  (open http://<pi-ip>:{args.port} from another device)")
    try:
        create_app(det).run(host=args.host, port=args.port, threaded=True)
    finally:
        det.stop()


if __name__ == "__main__":
    main()
