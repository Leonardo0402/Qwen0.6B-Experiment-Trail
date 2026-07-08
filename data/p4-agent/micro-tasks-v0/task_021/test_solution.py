from solution import classify


def test_excellent():
    assert classify(95) == "excellent"


def test_pass():
    assert classify(70) == "pass"


def test_fail():
    assert classify(50) == "fail"
