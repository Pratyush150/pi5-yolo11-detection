from person_detector import PersonDetector


def test_finds_single_person(image):
    people = PersonDetector(video=False).detect(image("skier"))
    assert len(people) == 1 and people[0]["conf"] > 0.8
    assert people[0]["label"] == "person"


def test_finds_partly_hidden_person(image):
    assert len(PersonDetector(video=False).detect(image("baseball"))) == 2


def test_no_person_in_photo_without_people(image):
    assert PersonDetector(video=False).detect(image("cats")) == []


def test_video_mode_needs_three_frames(image):
    d, frame = PersonDetector(motion=False), image("skier")
    assert d.detect(frame) == [] and d.detect(frame) == []
    assert len(d.detect(frame)) == 1
