from solution import quotient


def test_quotient():
    assert quotient(10, 2) == 5
    assert quotient(0, 5) == 0
    assert quotient(7, 7) == 1
