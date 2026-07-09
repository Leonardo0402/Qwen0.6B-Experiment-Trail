from solution import bmi_category


def test_obese():
    assert bmi_category(35) == "obese"


def test_overweight():
    assert bmi_category(27) == "overweight"


def test_normal():
    assert bmi_category(22) == "normal"


def test_underweight():
    assert bmi_category(17) == "underweight"
