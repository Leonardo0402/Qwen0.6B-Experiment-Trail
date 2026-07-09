def normalize(values):
    total = sum(values)
    if total == 0:
        return values
    return [v / total for v in values]
