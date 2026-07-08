import pytest
from solution import safe_get


def test_normal():
    assert safe_get([10, 20, 30], 1) == 20


def test_zero_idx():
    assert safe_get([10, 20, 30], 0) == 10


def test_negative_idx():
    with pytest.raises(ValueError):
        safe_get([10, 20, 30], -1)
