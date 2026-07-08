def min_of(a, b):
    return a if a < b else b


def max_of(a, b):
    return a if a < b else b  # BUG: should be a if a > b else b


def abs_val(n):
    return n if n >= 0 else -n
