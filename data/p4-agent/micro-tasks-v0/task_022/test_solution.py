from solution import grade_letter


def test_a():
    assert grade_letter(95) == "A"


def test_b():
    assert grade_letter(85) == "B"


def test_c():
    assert grade_letter(75) == "C"


def test_d():
    assert grade_letter(65) == "D"


def test_f():
    assert grade_letter(50) == "F"
