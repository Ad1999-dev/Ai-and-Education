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
    for gram_size in range(1, n + 1):
        dictionary = {}
        for k in range(len(document) - gram_size + 1):
            gram = document[k:k + gram_size]
            if gram in dictionary:
                dictionary[gram] += 1
            else:
                dictionary[gram] = 1
        tables.append(dictionary)
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
    if len(sequence) + 1 > len(tables):
        sequence = sequence[-(len(tables) - 1):] 
    
    n = len(sequence) + 1
    table_for_prefix = tables[n - 2]
    table_for_full = tables[n - 1]
    full_ngram = sequence + char
    
    prefix_count = table_for_prefix.get(sequence, 0)
    full_count = table_for_full.get(full_ngram, 0)

    if prefix_count == 0:
        return 0.0
    return full_count / prefix_count

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
    probability = 0
    best = None

    for char in vocabulary:
        char_prob = calculate_probability(sequence, char, tables)
        if char_prob > probability:
            probability = char_prob
            best = char

    return best 
