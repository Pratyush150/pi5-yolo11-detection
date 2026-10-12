import time

import cv2
import numpy as np

from app import Detector, create_app


def wait_for(client, status, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get("/api/state").get_json()
        if s["status"] == status and (status != "live" or s["counts"]):
            return s
        time.sleep(0.2)
    raise AssertionError(f"never reached {status}: {s}")


def test_ui_end_to_end(photo):
    det = Detector(str(photo("baseball")))
    client = create_app(det).test_client()
    try:
        assert client.get("/").status_code == 200
        s = wait_for(client, "live")
        assert s["counts"] == {"person": 2} and s["model"] == "YOLOX-tiny · person"

        r = client.post("/api/settings", json={"person_conf": 5, "confirm": False, "motion": False, "paused": True})
        assert r.get_json() == {"person_conf": 0.95, "motion": False, "confirm": False, "paused": True}

        snap = client.get("/api/snapshot.jpg")
        assert snap.status_code == 200 and snap.data[:2] == b"\xff\xd8"
        assert client.get("/api/raw.jpg").data[:2] == b"\xff\xd8"
    finally:
        det.stop()


def test_missing_camera_reports_status():
    det = Detector("/dev/does-not-exist")
    try:
        wait_for(create_app(det).test_client(), "no_camera")
    finally:
        det.stop()


def test_rotate_180_flips_still_image(tmp_path):
    img = np.zeros((40, 60, 3), np.uint8)
    img[:10, :10] = 255  # white marker top-left
    path = tmp_path / "marker.png"
    cv2.imwrite(str(path), img)
    det = Detector(str(path), rotate=180)
    _, still = det._open()
    det.stop()
    assert still[-1, -1].tolist() == [255, 255, 255] and still[0, 0].tolist() == [0, 0, 0]
