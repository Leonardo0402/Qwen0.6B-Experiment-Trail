def limit_value(x, lo, hi):
    if x < lo:
        return hi
    if x > hi:
        return lo
    return x + 1
