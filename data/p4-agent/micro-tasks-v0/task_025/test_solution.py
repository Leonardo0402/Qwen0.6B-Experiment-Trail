from solution import temperature_label


def test_hot():
    assert temperature_label(35) == "hot"


def test_mild():
    assert temperature_label(20) == "mild"


def test_cold():
    assert temperature_label(5) == "cold"
