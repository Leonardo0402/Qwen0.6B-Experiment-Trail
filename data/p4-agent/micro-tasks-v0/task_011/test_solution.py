import pytest
from solution import safe_sqrt


def test_positive():
    assert safe_sqrt(4) == 2.0


def test_zero():
    assert safe_sqrt(0) == 0.0


def test_negative():
    with pytest.raises(ValueError):
        safe_sqrt(-1)
