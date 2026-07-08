def saturate(x, lo, hi):
    if x > hi:
        return lo
    if x < lo:
        return hi
    return x
