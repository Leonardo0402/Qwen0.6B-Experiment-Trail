from solution import pad_left


def test_pad_basic():
    assert pad_left("hi", 5) == "   hi"


def test_pad_exact():
    assert pad_left("hello", 5) == "hello"


def test_pad_one():
    assert pad_left("ab", 3) == " ab"
