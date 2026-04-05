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
    
    # Generate frequency tables for n-grams from 1 to n
    for i in range(1, n + 1):
        freq_table = defaultdict(int)

        # Create n-grams
        for j in range(len(document) - i + 1):
            ngram = document[j:j + i]
            freq_table[ngram] += 1

        # Convert defaultdict to a regular dict for easier usage
        frequency_tables.append(dict(freq_table))

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
    if n == 0 or n > len(tables):
        return 0
    
    table = tables[n - 1]
    if sequence in table:
        total_count = sum(table[sequence].values())
        char_count = table[sequence][char]
        return char_count / total_count if total_count > 0 else 0
    else:
        return 0



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

    # Return the best character, default to 'a' if no character has a probability > 0
    return best_char if best_char else 'a'
