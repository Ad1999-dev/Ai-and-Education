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
    
    # Process n-grams from 1 to n
    for i in range(1, n + 1):
        frequency_table = defaultdict(lambda: defaultdict(int))

        # Generate n-grams
        for j in range(len(document) - i):
            # Extract the n-gram
            n_gram = document[j:j + i]
            next_char = document[j + i] if j + i < len(document) else ""

            # Update frequency table
            frequency_table[n_gram][next_char] += 1

        # Convert defaultdict back to a regular dict for cleaner output
        frequency_tables.append(dict(frequency_table))
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
     n = len(sequence)
    
    # Check if n is within the bounds of the given frequency tables
    if n > len(tables) or n == 0:
        return 0.0

    # Get the corresponding frequency table for the given sequence length
    freq_table = tables[n - 1]  # n - 1 because of 0-based index

    # Total number of occurrences of the sequence
    total_sequence_count = sum(freq_table.get(sequence, {}).values())
    
    # Count of occurrences of the character following the sequence
    char_count = freq_table.get(sequence, {}).get(char, 0)

    # Calculate probability
    if total_sequence_count == 0:
        return 0.0  # If the sequence was never seen, probability is 0

    probability = char_count / total_sequence_count
    return probability


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
    max_prob = 0.0
    predicted_char = None

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        
        if prob > max_prob:
            max_prob = prob
            predicted_char = char
    return predicted_char
