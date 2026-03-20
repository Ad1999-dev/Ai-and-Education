def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    frequency_table = [{} for _ in range(n)]

    for i in range(len(document)):
        for j in range(1, n+1):
            if i+j <= len(document):
                seq = tuple(document[i:i + j])
                table = frequency_table[j-1]

                if seq in table:
                    table[seq] += 1
                else:
                    table[seq] = 1
    return frequency_table


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
    seq_len = len(sequence)
    seq_tuple = tuple(sequence)

    if (seq_len >= len(tables)):
        seq_tuple = seq_tuple[-(len(tables) - 1):]
    else:
        seq_len += 1
    
    numerator = tables[seq_len - 1].get(seq_tuple + char, 0)
    denominator = tables[seq_len - 2].get(seq_tuple, 0)

    if denominator == 0:
        return 0
    else:
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
    probabilities = {}
    n = len(tables)

    if n == 1:
        return max(tables[0], key=tables[0].get)[0]

    for char in vocabulary:
        probability = calculate_probability(sequence, char, tables)
        probabilities[char] = probability

    if probabilities:
        return max(probabilities, key=probabilities.get)[0]
    return ""
