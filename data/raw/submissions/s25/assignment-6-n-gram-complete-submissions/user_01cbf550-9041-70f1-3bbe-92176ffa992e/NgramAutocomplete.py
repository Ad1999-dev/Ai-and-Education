def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    tables = []

    for i in range(n):
        cur_n = i + 1
        cur_table = {}

        for j in range(len(document) - cur_n + 1):
            ngram = document[j : j + cur_n]
            cur_table[ngram] = cur_table.get(ngram, 0) + 1

        tables.append(cur_table)

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

    n = len(tables)
    context_length = min(len(sequence), n - 1)
    context = sequence[-context_length:] if context_length > 0 else ""

    num_table = tables[context_length]
    ngram = context + char
    numerator = num_table.get(ngram, 0)

    if context_length > 0:
        den_table = tables[context_length - 1]
        denominator = den_table.get(context, 0)
    else:
        denominator = sum(tables[0].values())

    if denominator == 0:
        return 0.0
    return numerator / denominator


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
    best_prob = 0.0

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > best_prob or best_char is None:
            best_prob = prob
            best_char = char

    return best_char