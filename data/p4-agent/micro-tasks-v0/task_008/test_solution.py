from solution import exponent


def test_exponent():
    assert exponent(2, 3) == 8
    assert exponent(5, 0) == 1
    assert exponent(9, 1) == 9
