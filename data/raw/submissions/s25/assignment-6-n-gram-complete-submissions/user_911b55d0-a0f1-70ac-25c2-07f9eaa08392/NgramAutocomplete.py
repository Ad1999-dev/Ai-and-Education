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
        table = {}
        for j in range(len(document) - i):
            ngram = document[j:j + i + 1]
            if ngram not in table:
                table[ngram] = 0
            table[ngram] += 1
        tables.append(table)

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
    order = len(sequence) + 1

    if order > n:
        sequence = sequence[-(n - 1):]
        order = n

    full_ngram = sequence + char
    numerator_table = tables[order - 1]
    denominator_table = tables[order - 2]

    numerator = numerator_table.get(full_ngram, 0)
    denominator = denominator_table.get(sequence, 0)

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
    max_prob = -1
    predicted_char = None
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            predicted_char = char
    if predicted_char is None:
        raise ValueError("No valid prediction found. Check the input sequence and vocabulary.")
    return predicted_char
