from solution import clip_to_range


def test_below():
    assert clip_to_range(-5, 0, 10) == 0


def test_above():
    assert clip_to_range(15, 0, 10) == 10


def test_in_range():
    assert clip_to_range(5, 0, 10) == 5


def test_at_bounds():
    assert clip_to_range(0, 0, 10) == 0
    assert clip_to_range(10, 0, 10) == 10
