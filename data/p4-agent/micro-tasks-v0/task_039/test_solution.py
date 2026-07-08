from solution import fractionalize


def test_basic():
    result = fractionalize([10, 20, 30])
    assert abs(result[0] - 10.0/3.0) < 1e-9
    assert abs(result[1] - 20.0/3.0) < 1e-9
    assert abs(result[2] - 30.0/3.0) < 1e-9


def test_empty():
    assert fractionalize([]) == []


def test_single():
    assert fractionalize([5]) == [5.0]


def test_pair():
    result = fractionalize([4, 6])
    assert abs(result[0] - 2.0) < 1e-9
    assert abs(result[1] - 3.0) < 1e-9
