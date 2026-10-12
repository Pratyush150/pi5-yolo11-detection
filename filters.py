"""Video post-processing that makes person detection more reliable.

Detections are dicts: {"label": "person", "conf": float, "box": [x1, y1, x2, y2]}.
"""
import math
import time
from collections import deque

import cv2
import numpy as np


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    inter = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    return inter / (area(a) + area(b) - inter + 1e-9)


class Confirmer:
    """Report an object only after it was seen in `min_hits` of the last `window` frames,
    and keep reporting it through up to `hold` missed frames. Removes one-frame
    false alarms and flicker."""

    def __init__(self, min_hits=3, window=5, hold=2, match_iou=0.3):
        self.min_hits, self.window, self.hold, self.match_iou = min_hits, window, hold, match_iou
        self.tracks = []

    def update(self, objects):
        unmatched = list(range(len(self.tracks)))
        for o in sorted(objects, key=lambda o: -o["conf"]):
            best, best_iou = None, self.match_iou
            for i in unmatched:
                v = iou(self.tracks[i]["obj"]["box"], o["box"])
                if v >= best_iou:
                    best, best_iou = i, v
            if best is None:
                self.tracks.append({"obj": o, "hits": deque([True], maxlen=self.window), "missed": 0})
            else:
                t = self.tracks[best]
                t["obj"], t["missed"] = o, 0
                t["hits"].append(True)
                unmatched.remove(best)
        for i in unmatched:
            t = self.tracks[i]
            t["hits"].append(False)
            t["missed"] += 1
        self.tracks = [t for t in self.tracks if any(t["hits"]) and t["missed"] <= self.hold]
        return [t["obj"] for t in self.tracks if sum(t["hits"]) >= self.min_hits]

    def reset(self):
        self.tracks = []


class MotionGate:
    """Accept a person box only once something has moved inside it.

    Static look-alikes (a chair back, a coat, a wig) never move, so they are never accepted.
    A real person moves at least when they come into view; after that their box is remembered
    and followed for `memory_s` seconds even if they sit completely still.

    Motion = difference from a slowly updated background (time constant `tau_s`), so it works the
    same at 5 or 30 FPS. A box counts as moving when at least `min_moving` of its pixels changed by more
    than `diff_thresh` grey levels (tuned on the Pi HQ camera: a nearly still person is accepted within
    ~1.5 s, while sensor noise on empty background never triggered it).
    """

    def __init__(self, min_moving=0.01, diff_thresh=10, tau_s=3.0, memory_s=10.0, follow_iou=0.3, scale=0.25):
        self.min_moving, self.diff_thresh, self.tau_s = min_moving, diff_thresh, tau_s
        self.memory_s, self.follow_iou, self.scale = memory_s, follow_iou, scale
        self.reset()

    def reset(self):
        self.background, self.last_t, self.verified = None, None, []  # verified: [box, last_seen]

    def update(self, frame, people, now=None):
        now = time.monotonic() if now is None else now
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                                                         interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
        if self.background is None or self.background.shape != small.shape:  # first frame, or new resolution
            self.background, self.last_t = small, now
            return [p for p in people if any(iou(v[0], p["box"]) >= self.follow_iou for v in self.verified)]
        diff = small - self.background
        diff -= np.median(diff)  # a light switching on or auto-exposure shifts everything; that is not motion
        moving = np.abs(diff) > self.diff_thresh
        alpha = 1 - math.exp(-max(now - self.last_t, 0.0) / self.tau_s)
        cv2.accumulateWeighted(small, self.background, alpha)
        self.last_t = now

        self.verified = [v for v in self.verified if now - v[1] <= self.memory_s]
        accepted = []
        for p in people:
            x1, y1, x2, y2 = (int(round(c * self.scale)) for c in p["box"])
            x1, y1 = max(x1, 0), max(y1, 0)
            region = moving[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
            known = [v for v in self.verified if iou(v[0], p["box"]) >= self.follow_iou]
            if region.size and region.mean() >= self.min_moving or known:
                for v in known:
                    self.verified.remove(v)
                self.verified.append([p["box"], now])
                accepted.append(p)
        return accepted
