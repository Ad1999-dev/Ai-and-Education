from collections import defaultdict
def create_frequency_tables(document, n):
    document = document.replace("\n", " ")
    length = len(document)
    tables = [defaultdict(int) for _ in range(n)]

    for i in range(length):
        for j in range(1, n+1):
            if i + j <= length:
                ngram = document[i:i+j]
                tables[j-1][ngram] += 1

    return tables


def calculate_probability(sequence, char, tables):

    n = len(tables)
    max_order = min(n, len(sequence) + 1)

    for k in range(max_order, 0, -1):
        prefix = sequence[-(k-1):] if k > 1 else ''
        full_ngram = prefix + char
        ngram_table = tables[k-1]

        full_count = ngram_table.get(full_ngram, 0)
        prefix_total = sum(v for k2, v in ngram_table.items() if k2.startswith(prefix))

        if prefix_total > 0:
            return full_count / prefix_total

    return 1e-6


def predict_next_char(sequence, tables, vocabulary):

    best_char = None
    best_prob = -1

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > best_prob:
            best_prob = prob
            best_char = char

    return best_char

