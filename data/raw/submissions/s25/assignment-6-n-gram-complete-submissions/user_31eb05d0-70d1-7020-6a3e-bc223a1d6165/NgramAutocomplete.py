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

    doc_length = len(document)

    for i in range(doc_length):
        for j in range(n):
            if i - j < 0:
                continue

            prev_chars = document[i - j:i]
            current_char = document[i]

            tables[j][current_char][prev_chars] += 1

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
    n = len(sequence) + 1 
    if n > len(tables):
        return 0

    table = tables[n - 1]
    numerator = table[char][sequence]
    
    denominator = sum(count for next_char in table for seq, count in table[next_char].items() if seq == sequence)

    if denominator == 0:
        return 0

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
    max_prob = 0
    best_char = None
    
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char
    
    return best_char if best_char is not None else ' '
