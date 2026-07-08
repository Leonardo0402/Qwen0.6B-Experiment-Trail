def fractionalize(values):
    n = len(values)
    if n == 0:
        return values
    return [v / (n - 1) for v in values]
