def standardize(values):
    mean = sum(values) / len(values)
    if mean == 0:
        return values
    return [v - mean for v in values]
