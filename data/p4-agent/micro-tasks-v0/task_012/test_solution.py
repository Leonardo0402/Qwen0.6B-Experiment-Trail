import pytest
from solution import safe_divide


def test_normal():
    assert safe_divide(10, 2) == 5


def test_zero_numerator():
    assert safe_divide(0, 5) == 0


def test_zero_denominator():
    with pytest.raises(ValueError):
        safe_divide(1, 0)
