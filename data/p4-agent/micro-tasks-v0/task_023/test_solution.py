from solution import speed_label


def test_fast():
    assert speed_label(75) == "fast"


def test_medium():
    assert speed_label(50) == "medium"


def test_slow():
    assert speed_label(20) == "slow"
