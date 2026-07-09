from solution import echo


def test_echo_basic():
    assert echo("ab", 3) == "ababab"


def test_echo_zero():
    assert echo("x", 0) == ""


def test_echo_one():
    assert echo("y", 1) == "y"
