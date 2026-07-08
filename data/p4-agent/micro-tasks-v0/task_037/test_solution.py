from solution import standardize


def test_basic():
    result = standardize([1, 2, 3])
    assert abs(sum(result)) < 1e-9


def test_zero_mean():
    assert standardize([1, -1]) == [0.0, 0.0]


def test_single():
    assert standardize([5]) == [0.0]


def test_negative():
    result = standardize([-2, -4, -6])
    assert abs(sum(result)) < 1e-9
