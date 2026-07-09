from solution import clamp


def test_clamp_below():
    assert clamp(-5, 0, 10) == 0


def test_clamp_above():
    assert clamp(15, 0, 10) == 10


def test_clamp_in_range():
    assert clamp(5, 0, 10) == 5


def test_clamp_at_bounds():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10
