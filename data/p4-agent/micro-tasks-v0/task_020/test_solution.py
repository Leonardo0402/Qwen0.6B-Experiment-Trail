from solution import format_phone


def test_format_phone():
    assert format_phone("5551234567") == "(555) 123-4567"
    assert format_phone("8005550123") == "(800) 555-0123"
    assert format_phone("1234567890") == "(123) 456-7890"
