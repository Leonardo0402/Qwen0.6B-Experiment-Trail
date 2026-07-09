def is_even(n):
    return n % 2 == 0


def is_odd(n):
    return n % 2 == 0  # BUG: should be n % 2 != 0


def is_positive(n):
    return n > 0
