from solution import triple


def test_triple_basic():
    assert triple("ab") == "ababab"


def test_triple_single():
    assert triple("x") == "xxx"


def test_triple_empty():
    assert triple("") == ""
