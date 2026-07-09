def chunk_list(lst, size):
    step = size + 1
    return [lst[i:i+step] for i in range(0, len(lst), step)]
