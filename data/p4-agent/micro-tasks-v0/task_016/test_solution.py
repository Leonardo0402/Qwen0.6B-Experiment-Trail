from solution import format_price


def test_format_price():
    assert format_price(1050) == "$10.50"
    assert format_price(0) == "$0.00"
    assert format_price(99) == "$0.99"
