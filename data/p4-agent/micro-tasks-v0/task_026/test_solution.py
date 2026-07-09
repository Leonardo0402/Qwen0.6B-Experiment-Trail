from solution import repeat


def test_repeat_basic():
    assert repeat("ab", 3) == "ababab"


def test_repeat_zero():
    assert repeat("x", 0) == ""


def test_repeat_one():
    assert repeat("y", 1) == "y"
