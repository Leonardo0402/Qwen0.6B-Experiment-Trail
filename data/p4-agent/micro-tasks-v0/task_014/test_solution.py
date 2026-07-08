import pytest
from solution import safe_inverse


def test_one():
    assert safe_inverse(1) == 1.0


def test_two():
    assert safe_inverse(2) == 0.5


def test_zero():
    with pytest.raises(ValueError):
        safe_inverse(0)
