def rescale(values, new_min, new_max):
    old_min = min(values)
    old_max = max(values)
    if old_max == old_min:
        return values
    return [(v - old_min) / (old_max - old_min) * (new_max - new_min) + new_min for v in values]
