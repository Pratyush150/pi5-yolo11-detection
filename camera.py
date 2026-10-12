"""Camera input: Raspberry Pi camera (Picamera2), USB webcam or video file, newest frame only."""
import threading

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def picam_available() -> bool:
    try:
        from picamera2 import Picamera2
        return len(Picamera2.global_camera_info()) > 0
    except Exception:
        return False


class LatestFrame:
    """Reads frames in a background thread and keeps only the newest one.

    rotate=180 turns the image for an upside-down camera (done by the Pi camera
    hardware; software rotation for webcams and files).
    """

    def __init__(self, source, rotate=0):
        if rotate not in (0, 180):
            raise ValueError("rotate must be 0 or 180")
        self.picam, self.rotate = None, rotate
        if source == "auto":
            source = "picam" if picam_available() else "0"
        if source == "picam":
            from libcamera import Transform
            from picamera2 import Picamera2
            self.picam = Picamera2()
            self.picam.configure(self.picam.create_preview_configuration(
                main={"format": "RGB888", "size": (1280, 720)},
                transform=Transform(hflip=1, vflip=1) if rotate == 180 else Transform()))
            self.picam.start()
            self.is_file = False
        else:
            src = int(source) if str(source).isdigit() else source
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise SystemExit(f"Could not open source: {source}")
            self.is_file = not isinstance(src, int)
        self.frame, self.ok, self.stopped = None, True, False
        self.seq, self._last_seq = 0, 0  # frame counter, so read() never returns the same frame twice
        self.lock = threading.Condition()
        if not self.is_file:
            threading.Thread(target=self._loop, daemon=True).start()

    def _grab(self):
        if self.picam is not None:
            return True, self.picam.capture_array()  # RGB888 => BGR order in memory
        ok, frame = self.cap.read()
        if ok and self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return ok, frame

    def _loop(self):
        while not self.stopped:
            ok, frame = self._grab()
            with self.lock:
                self.ok, self.frame = ok, frame
                self.seq += 1
                self.lock.notify_all()
            if not ok:
                break

    def read(self):
        if self.is_file:  # process every frame of a video file
            return self._grab()
        with self.lock:
            # wait for a frame newer than the last one returned (camera rate caps the detector rate)
            self.lock.wait_for(lambda: self.seq != self._last_seq or not self.ok or self.stopped, timeout=2)
            self._last_seq = self.seq
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        if self.picam is not None:
            self.picam.stop()
            self.picam.close()  # free the camera so it can be reopened
        else:
            self.cap.release()
