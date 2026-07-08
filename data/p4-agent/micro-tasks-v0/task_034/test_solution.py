from solution import bound_value


def test_below():
    assert bound_value(-5, 0, 10) == 0


def test_above():
    assert bound_value(15, 0, 10) == 10


def test_in_range():
    assert bound_value(5, 0, 10) == 5


def test_at_bounds():
    assert bound_value(0, 0, 10) == 0
    assert bound_value(10, 0, 10) == 10
