def create_frequency_tables(document, n):
    tables: list[dict[str,int]] = []
    L = len(document)
    for k in range(1, n+1):
        freq: dict[str,int] = {}
        for i in range(L - k + 1):
            seq = document[i : i + k]
            freq[seq] = freq.get(seq, 0) + 1
        tables.append(freq)
    return tables


def calculate_probability(sequence, char, tables):
    max_n = len(tables)

    if len(sequence) >= max_n:
        sequence = sequence[-(max_n-1):]

    k = len(sequence)
    joint = sequence + char

    num = tables[k].get(joint, 0)

    if k == 0:
        total = sum(tables[0].values())
        return (num / total) if total > 0 else 0.0

    denom = tables[k-1].get(sequence, 0)
    return (num / denom) if denom > 0 else 0.0


def predict_next_char(sequence, tables, vocabulary):
    best_char = ''
    best_score = -1.0
    for c in vocabulary:
        p = calculate_probability(sequence, c, tables)
        if p > best_score:
            best_score, best_char = p, c
    return best_char
