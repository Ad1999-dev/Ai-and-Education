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
    tables = [defaultdict(int) for _ in range(n)]
    length = len(document)
    for i in range(length):
        for order in range(1, n + 1):
            if i + order <= length:
                seq = document[i : i + order]
                tables[order - 1][seq] += 1
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
    length = len(sequence)
    if length > max_order - 1:
        sequence = sequence[-(max_order - 1):]
        length = max_order - 1
    joint = tables[length].get(sequence + char, 0)
    if length == 0:
        total = sum(tables[0].values())
    else:
        total = sum(
            tables[length].get(sequence + c, 0)
            for c in tables[0].keys()
        )
    return joint / total if total > 0 else 0.0


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
    best_prob = -1.0
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > best_prob:
            best_prob = prob
            best_char = char
    return best_char
