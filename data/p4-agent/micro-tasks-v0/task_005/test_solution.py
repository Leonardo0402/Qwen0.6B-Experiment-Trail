from solution import to_celsius, to_fahrenheit, to_kelvin


def test_to_celsius():
    assert to_celsius(32) == 0.0


def test_to_fahrenheit():
    assert to_fahrenheit(0) == 32.0


def test_to_kelvin():
    assert to_kelvin(0) == 273.15
