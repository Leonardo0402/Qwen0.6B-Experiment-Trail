from solution import remainder


def test_remainder():
    assert remainder(10, 3) == 1
    assert remainder(8, 4) == 0
    assert remainder(7, 5) == 2
