from solution import compute


def test_compute():
    assert compute(3, 4) == 17
    assert compute(0, 5) == 15
    assert compute(1, 1) == 12
