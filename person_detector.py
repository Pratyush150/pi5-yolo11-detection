"""Person detection API for applications.

    from person_detector import PersonDetector

    detector = PersonDetector()             # loads the model once (~0.2 s)
    people = detector.detect(frame)         # frame: BGR numpy image (OpenCV / Picamera2 "RGB888")
    for p in people:
        print(p["conf"], p["box"])          # box = [x1, y1, x2, y2] in frame pixels

detect() runs YOLOX-tiny on the CPU (~30 ms per 1280x720 frame on a Raspberry Pi 5) and keeps people at or
above `conf`. For video (the default) it also:
  - ignores person-like boxes where nothing has ever moved (a chair back, a coat, a wig): in a test on 250
    COCO photos of furniture without people, the model alone called something a person in 4 of them, and
  - reports a person only after they appeared in 3 of the last 5 frames.
Use one PersonDetector per camera stream, and call detect() on every frame. For single photos pass
video=False. Not thread-safe: call detect() from one thread per instance.
"""
from filters import Confirmer, MotionGate
from yolox_backend import MODEL_PATH, YoloxBackend


class PersonDetector:
    def __init__(self, conf=0.45, video=True, motion=True, confirm=True, threads=3, model_path=MODEL_PATH):
        """conf: person confidence threshold (0.45 still catches partly hidden people).
        video: frames come from one camera stream; enables the two checks below (False for photos).
        motion: (video) only accept a person once something moved inside their box.
        confirm: (video) only report a person seen in 3 of the last 5 frames.
        threads: CPU threads for the model (3 leaves a core free on a 4-core Pi)."""
        self.conf, self.video, self.motion, self.confirm = conf, video, motion, confirm
        self.model = YoloxBackend(model_path, threads=threads)
        self.model.load()
        self._gate, self._confirmer = MotionGate(), Confirmer()

    @property
    def label(self):
        return self.model.label

    def detect(self, frame, conf=None, video=None, motion=None, confirm=None):
        """People in `frame`, as a list of {"label": "person", "conf": float, "box": [x1, y1, x2, y2]}.
        The keyword arguments override the constructor settings for this call."""
        pick = lambda value, default: default if value is None else value  # noqa: E731
        people = self.model.infer(frame, pick(conf, self.conf))
        if not pick(video, self.video):
            return people
        gated = self._gate.update(frame, people)  # always called, so it keeps learning the background
        if pick(motion, self.motion):
            people = gated
        if pick(confirm, self.confirm):
            people = self._confirmer.update(people)
        return people

    def reset(self):
        """Forget the scene and confirmation history, e.g. when switching to a different camera."""
        self._gate.reset()
        self._confirmer.reset()
