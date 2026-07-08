from solution import divide, modulo, power


def test_divide():
    assert divide(10, 2) == 5


def test_modulo():
    assert modulo(10, 3) == 1


def test_power():
    assert power(2, 3) == 8
