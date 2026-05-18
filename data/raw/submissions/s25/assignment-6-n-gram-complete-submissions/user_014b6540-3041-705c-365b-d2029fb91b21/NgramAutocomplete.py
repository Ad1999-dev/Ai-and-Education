from collections import defaultdict

def create_frequency_tables(document, n):
    """
    This function constructs a list of n frequency tables for an n-gram model,
    each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]

    # Build the frequency tables
    for i in range(len(document)):
        for k in range(1, n + 1):
            if i >= k:
                prev_sequence = document[i - k:i]
                next_char = document[i]
                frequency_tables[k - 1][prev_sequence][next_char] += 1

    return [dict(table) for table in frequency_tables]  

def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    n = len(tables)
    seq_length = len(sequence)

    # We need at least 1 character of context to calculate the probability
    if seq_length == 0:
        return 0.0
    
    # Determine the max context length we can use
    context_length = min(seq_length, n)

    # Iterate over the context lengths from max to min
    for l in range(context_length, 0, -1):
        context = sequence[-l:]  # last l characters of the sequence
        if context in tables[l-1]:  # l is 1-indexed, tables are 0-indexed
            return tables[l-1][context].get(char, 0.0)  # Get the probability of the char
    
    return 0.0  # If no context found

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
    
    return predicted_char if predicted_char is not None else 'a'  # default to 'a' if nothing found
