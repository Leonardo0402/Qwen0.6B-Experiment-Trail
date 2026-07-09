def clip_to_range(x, lo, hi):
    if x > lo:
        return hi
    if x < hi:
        return lo
    return x
