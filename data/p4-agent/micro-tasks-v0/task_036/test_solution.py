from solution import normalize


def test_basic():
    result = normalize([1, 2, 3])
    assert abs(sum(result) - 1.0) < 1e-9


def test_zero_sum():
    assert normalize([0, 0, 0]) == [0.0, 0.0, 0.0]


def test_single():
    assert normalize([5]) == [1.0]


def test_negative():
    result = normalize([1, -1])
    assert result == [0.0, 0.0]
