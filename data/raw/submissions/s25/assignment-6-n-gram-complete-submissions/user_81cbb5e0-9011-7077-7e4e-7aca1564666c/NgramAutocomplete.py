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

    for i in range(1, n+1):
        table = {}

        for j in range(len(document) - i + 1):
            ngram = document[j: j + i]
            if ngram in table:
                table[ngram] += 1
            else:
                table[ngram] = 1
            
        tables.append(table)

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

    n = len(sequence) + 1  # We need to look at the (length of sequence + 1)th table
    if n > len(tables) or n == 0:
        return 0.0  # If n is out of bounds for the list of tables

    # Get the relevant frequency table
    freq_table1 = tables[n - 2]
    freq_table2 = tables[n - 1]  # n-1 because list is 0-indexed

    # Count of the sequence
    sequence_count = freq_table1.get(sequence, 0)

    # Count of the sequence + char (sequence followed by char)
    combined_ngram = sequence + char
    combined_count = freq_table2.get(combined_ngram, 0)

    # Calculate the probability
    if sequence_count == 0:
        return 0.0  # Avoid division by zero

    probability = combined_count / sequence_count

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

    max_probability = 0  # Initialize to 0 since probability can be 0 for unpredicted cases
    predicted_char = ''

    # Sort vocabulary once to avoid sorting in every iteration
    sorted_vocabulary = sorted(vocabulary)

    for char in sorted_vocabulary:
        probability = calculate_probability(sequence, char, tables)

        if probability > max_probability:
            max_probability = probability
            predicted_char = char

    # If no valid prediction, return the most frequent character (first in sorted list)
    if max_probability == 0:
        predicted_char = sorted_vocabulary[0]

    return predicted_char

