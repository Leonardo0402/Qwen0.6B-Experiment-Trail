def bound_value(x, lo, hi):
    if x >= lo:
        return lo
    if x <= hi:
        return hi
    return x
