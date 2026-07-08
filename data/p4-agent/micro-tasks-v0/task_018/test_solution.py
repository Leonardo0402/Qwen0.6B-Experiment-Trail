from solution import format_date


def test_format_date():
    assert format_date(2026, 7, 8) == "2026-07-08"
    assert format_date(2000, 1, 1) == "2000-01-01"
    assert format_date(1999, 12, 31) == "1999-12-31"
