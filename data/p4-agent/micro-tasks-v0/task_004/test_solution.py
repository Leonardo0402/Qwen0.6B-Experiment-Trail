from solution import min_of, max_of, abs_val


def test_min_of():
    assert min_of(3, 7) == 3


def test_max_of():
    assert max_of(3, 7) == 7


def test_abs_val():
    assert abs_val(-5) == 5
