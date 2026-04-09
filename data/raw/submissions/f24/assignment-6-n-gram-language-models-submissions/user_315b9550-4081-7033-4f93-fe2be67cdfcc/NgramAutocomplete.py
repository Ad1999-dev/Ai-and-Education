def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables = []
    for _ in range(n):
        frequency_tables.append({})

    for i in range(len(document)):
        for gram in range(1, n + 1):
            if i + gram <= len(document):
                if gram == 1:
                    ngram = document[i]
                else:
                    ngram = tuple(document[i:i + gram])
                
                if ngram in frequency_tables[gram - 1]:
                    frequency_tables[gram - 1][ngram] += 1
                else:
                    frequency_tables[gram - 1][ngram] = 1
    return frequency_tables


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
        n = len(tables)

    final_sequence = sequence[-(n - 1):] if len(sequence) >= (n - 1) else sequence
    ngram = tuple(final_sequence) + (char,)
    ngram_count = tables[n - 1].get(ngram, 0)

    if n > 1:  
        prefix_count = tables[n - 2].get(tuple(final_sequence), 0)
    else:  
        prefix_count = sum(tables[0].values())

    if prefix_count == 0:
        return 0.0
    return ngram_count / prefix_count


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
    n = len(tables)  
    final_sequence = sequence[-(n - 1):] if len(sequence) >= (n - 1) else sequence

    max_prob = 0
    predicted_char = None

    for char in vocabulary:
        prob = calculate_probability(final_sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            predicted_char = char

    return predicted_char if predicted_char is not None else ' '
