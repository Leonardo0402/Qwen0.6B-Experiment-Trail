def to_celsius(f):
    return (f + 32) * 5.0 / 9.0  # BUG: should be (f - 32) * 5.0 / 9.0


def to_fahrenheit(c):
    return c * 9.0 / 5.0 + 32.0


def to_kelvin(c):
    return c + 273.15
