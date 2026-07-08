from solution import proportion


def test_basic():
    assert abs(proportion([10, 20, 30], 1) - 20.0/60.0) < 1e-9


def test_zero_total():
    assert proportion([5, -5, 0], 0) == 0.0


def test_first():
    assert abs(proportion([1, 2, 3], 0) - 1.0/6.0) < 1e-9


def test_negative_total():
    assert abs(proportion([1, -1, 2], 2) - 2.0/2.0) < 1e-9
