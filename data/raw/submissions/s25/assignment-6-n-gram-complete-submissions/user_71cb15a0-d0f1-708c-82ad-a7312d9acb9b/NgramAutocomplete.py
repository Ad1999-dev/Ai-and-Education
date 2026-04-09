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

    for i in range(len(document)):
        for size in range(1, n + 1):
            if i + size <= len(document):
                g = document[i:i+size] 
                tables[size-1][g] += 1
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

    # cut sequence to only last n-1 characters
    if len(sequence) > n-1:
        sequence = sequence[-(n-1):]

    combined = sequence + char 
    if len(combined) == 1:
        freq_full = tables[0].get(combined, 0)
        total_chars = sum(tables[0].values())
        if total_chars == 0:
            return 0
        return freq_full / total_chars
    else:
        freq_full = tables[len(combined)-1].get(combined, 0)
        freq_prefix = tables[len(combined)-2].get(combined[:-1], 0)
        if freq_prefix == 0:
            return 0
        return freq_full / freq_prefix



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
    best_char = ' '

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char

    return best_char
