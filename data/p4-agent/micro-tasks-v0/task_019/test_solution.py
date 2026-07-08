from solution import format_name


def test_format_name():
    assert format_name("alice") == "Alice"
    assert format_name("BOB") == "Bob"
    assert format_name("charlie") == "Charlie"
