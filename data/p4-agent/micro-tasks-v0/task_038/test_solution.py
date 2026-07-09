from solution import rescale


def test_basic():
    result = rescale([1, 2, 3], 0, 10)
    assert abs(result[0] - 0.0) < 1e-9
    assert abs(result[1] - 5.0) < 1e-9
    assert abs(result[2] - 10.0) < 1e-9


def test_constant():
    assert rescale([5, 5, 5], 0, 10) == [0.0, 0.0, 0.0]


def test_single():
    assert rescale([7], 0, 1) == [0.0]


def test_negative_range():
    result = rescale([1, 2, 3], -1, 1)
    assert abs(result[0] + 1.0) < 1e-9
    assert abs(result[2] - 1.0) < 1e-9
