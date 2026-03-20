from collections import defaultdict

from utilities import print_table, read_file


def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    tables = [{} for _ in range(n)]
    document_length = len(document)

    for size in range(1, n + 1):
        for i in range(document_length - size + 1):
            ngram = document[i:i + size]  # Extract n-gram as a string
            if ngram in tables[size - 1]:
                tables[size - 1][ngram] += 1
            else:
                tables[size - 1][ngram] = 1

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

    n = len(sequence) + 1  # Length of n-gram including the character
    if n > len(tables):
        return 0.0  # Sequence length exceeds available n-grams

    full_ngram = sequence + char
    sequence_count = tables[n - 2].get(sequence, 0)
    full_ngram_count = tables[n - 1].get(full_ngram, 0)

    if sequence_count == 0:
        return 0.0
    return full_ngram_count / sequence_count


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

    max_prob = 0
    best_char = None

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char

    return best_char

