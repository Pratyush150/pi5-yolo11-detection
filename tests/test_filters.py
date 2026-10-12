import numpy as np

from filters import Confirmer, MotionGate


def person(box, conf=0.8):
    return {"label": "person", "conf": conf, "box": box}


def test_single_frame_blip_is_not_reported():
    c = Confirmer()
    assert c.update([person([0, 0, 100, 200])]) == []
    for _ in range(5):
        assert c.update([]) == []


def test_steady_detection_reported_after_three_frames_and_held_briefly():
    c = Confirmer()
    p = person([100, 100, 200, 400])
    assert c.update([p]) == [] and c.update([p]) == []
    assert len(c.update([p])) == 1                  # confirmed on 3rd frame
    assert len(c.update([])) == 1                   # held through one missed frame
    assert len(c.update([person([110, 105, 210, 405])])) == 1  # same track after a small move


def scene(rng, figure_x=None, brightness=0):
    """Grey room with sensor noise; optionally a dark figure at figure_x."""
    img = np.full((360, 640, 3), 120 + brightness, np.float32) + rng.normal(0, 2, (360, 640, 3))
    if figure_x is not None:
        img[100:300, figure_x:figure_x + 60] = 40 + brightness
    return np.clip(img, 0, 255).astype(np.uint8)


def test_static_lookalike_never_accepted():
    rng, g = np.random.default_rng(0), MotionGate()
    fake = person([300, 100, 360, 300])  # e.g. a chair back the model calls "person"
    frames = [scene(rng, figure_x=300) for _ in range(40)]
    assert all(g.update(f, [fake], now=i * 0.1) == [] for i, f in enumerate(frames))


def test_moving_person_accepted_and_kept_when_still():
    rng, g = np.random.default_rng(1), MotionGate()
    g.update(scene(rng), [], now=0)                     # empty room
    t = 0.0
    for x in range(100, 300, 20):                       # walks in
        t += 0.1
        out = g.update(scene(rng, figure_x=x), [person([x, 100, x + 60, 300])], now=t)
    assert out, "walking person not accepted"
    for _ in range(50):                                 # then stands still for 5 s
        t += 0.1
        out = g.update(scene(rng, figure_x=280), [person([280, 100, 340, 300])], now=t)
    assert out, "person lost after standing still"


def test_lights_turning_on_is_not_motion():
    rng, g = np.random.default_rng(2), MotionGate()
    fake = person([300, 100, 360, 300])
    g.update(scene(rng, figure_x=300), [fake], now=0)
    assert g.update(scene(rng, figure_x=300, brightness=40), [fake], now=0.1) == []
