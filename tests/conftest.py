import sys
import urllib.request
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = Path(__file__).resolve().parent / ".cache"

# COCO val2017 photos (CC BY 4.0): one skier, two baseball players (one partly hidden), two cats (no person)
COCO = {"skier": "000000000785", "baseball": "000000000872", "cats": "000000039769"}


@pytest.fixture(scope="session")
def photo():
    def load(name):
        path = CACHE / f"{COCO[name]}.jpg"
        if not path.exists():
            CACHE.mkdir(exist_ok=True)
            try:
                urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{path.name}", path)
            except OSError as e:
                pytest.skip(f"could not download test photo: {e}")
        return path
    return load


@pytest.fixture(scope="session")
def image(photo):
    return lambda name: cv2.imread(str(photo(name)))
