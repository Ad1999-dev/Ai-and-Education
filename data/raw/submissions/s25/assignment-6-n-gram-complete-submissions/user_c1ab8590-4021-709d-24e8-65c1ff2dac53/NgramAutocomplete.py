from collections import defaultdict


def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]

    for i, c in enumerate(document):
        for k in range(1, n + 1):
            if i >= k - 1:
                context = document[i - (k - 1) : i]
                tables[k - 1][c][context] += 1

    return tables



def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    max_order = len(tables)
    if len(sequence) >= max_order:
        context = sequence[-(max_order - 1) :]
        table_idx = max_order - 1
    else:
        context = sequence
        table_idx = len(context)

    count_ctx_char = tables[table_idx].get(char, {}).get(context, 0)
    total_ctx = sum(
        ctx_counts.get(context, 0)
        for ctx_counts in tables[table_idx].values()
    )
    if total_ctx == 0:
        return 0.0

    return count_ctx_char / total_ctx

def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character based on the given sequence.

    - **Parameters**:
        - `sequence`: The sequence used as input to predict the next character.
        - `tables`: The list of frequency tables.
        - `vocabulary`: The set of possible characters.
    
    - **Functionality**:
        - Calculates the probability of each possible next character in the vocabulary, using `calculate_probability()`.

    - **Returns**:
        - Returns the character with the maximum probability as the predicted next character.
    """
    best_char = None
    best_p = -1.0
    for c in vocabulary:
        p = calculate_probability(sequence, c, tables)
        if p > best_p:
            best_p = p
            best_char = c
    return best_char if best_char is not None else next(iter(vocabulary))