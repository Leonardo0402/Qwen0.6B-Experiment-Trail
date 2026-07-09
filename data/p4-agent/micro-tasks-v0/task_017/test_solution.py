from solution import format_percent


def test_format_percent():
    assert format_percent(0.5) == "50.0%"
    assert format_percent(1.0) == "100.0%"
    assert format_percent(0.0) == "0.0%"
