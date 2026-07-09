from solution import saturate


def test_below():
    assert saturate(-5, 0, 10) == 0


def test_above():
    assert saturate(15, 0, 10) == 10


def test_in_range():
    assert saturate(5, 0, 10) == 5


def test_at_bounds():
    assert saturate(0, 0, 10) == 0
    assert saturate(10, 0, 10) == 10
