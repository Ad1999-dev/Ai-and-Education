def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, 
    each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables = [{} for _ in range(n)]  # Initialize n frequency tables
    
    # Loop through each level of n-gram (from 1 to n)
    for i in range(len(document)):
        for j in range(1, n + 1):
            if i + j > len(document):
                break
            # Check if the current n-gram exists in the frequency table as a key
            ngram = document[i:i + j]
            if ngram not in frequency_tables[j - 1]:
                frequency_tables[j - 1][ngram] = 0
            frequency_tables[j - 1][ngram] += 1
    
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
    seq_length = len(sequence)
    
    # Ensure we do not exceed the number of tables
    if seq_length > len(tables):
        return 0.0  # Return 0 probability as a float
    
    # Handle the case for single character probabilities
    if seq_length == 0:  
        total_frequency = sum(tables[0].values())  # Total frequency of all single characters
        char_freq = tables[0].get(char, 0)  # Frequency of the character
        return char_freq / total_frequency if total_frequency > 0 else 0.0

    # Get the base frequency of the sequence
    base_freq = tables[seq_length - 1].get(sequence, 0)  # Default to 0 if the sequence is not found
    
    # Get frequency of the combined sequence (sequence + char)
    combined_freq = tables[seq_length].get(sequence + char, 0) if seq_length < len(tables) else 0
    
    # Compute and return the probability
    return (combined_freq / base_freq) if base_freq > 0 else 0.0


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

    return best_char if best_char else ''
