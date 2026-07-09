import pytest
from solution import safe_age


def test_normal():
    assert safe_age(25) == 25


def test_zero():
    assert safe_age(0) == 0


def test_negative():
    with pytest.raises(ValueError):
        safe_age(-5)
