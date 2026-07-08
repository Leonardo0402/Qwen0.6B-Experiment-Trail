from solution import limit_value


def test_below():
    assert limit_value(-5, 0, 10) == 0


def test_above():
    assert limit_value(15, 0, 10) == 10


def test_in_range():
    assert limit_value(5, 0, 10) == 5


def test_at_bounds():
    assert limit_value(0, 0, 10) == 0
    assert limit_value(10, 0, 10) == 10
