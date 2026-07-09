def proportion(values, idx):
    total = sum(values)
    if total == 0:
        return values[idx]
    return values[idx] / total
