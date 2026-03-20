from collections import defaultdict

def create_frequency_tables(document, n):
    """
    Builds n frequency tables:
    Table 1: f(char)
    Table 2: f(char | char)
    Table 3: f(char | char, char)
    ...
    Table n: f(char | n-1 char context)
    """
    tables = []

    for i in range(n):
        tables.append(defaultdict(lambda: defaultdict(int)))

    for i in range(len(document)):
        for k in range(1, n + 1):
            if i - k + 1 < 0:
                continue
            context = tuple(document[i - k + 1 : i])
            char = document[i]
            tables[k - 1][char][context] += 1

    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the conditional probability of `char` given `sequence`.
    """
    n = len(tables)
    context = tuple(sequence[-(n - 1):]) if n > 1 else ()

    numerator = tables[n - 1][char].get(context, 0)
    denominator = sum(tables[n - 1][c].get(context, 0) for c in tables[n - 1])

    if denominator == 0:
        return 0.0
    return numerator / denominator


def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the next character with the highest probability.
    """
    n = len(tables)
    context = tuple(sequence[-(n - 1):]) if n > 1 else ()

    best_char = None
    best_prob = -1

    for char_tuple in vocabulary:
        char = char_tuple[0]
        prob = calculate_probability(sequence, char, tables)
        if prob > best_prob:
            best_prob = prob
            best_char = char

    return best_char
