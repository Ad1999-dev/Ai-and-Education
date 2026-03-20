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
    frequency_tables = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]

    # Iterate through the document to build the n-grams
    for i in range(len(document) - n + 1):
        n_gram = document[i:i + n]  # Extract the n-gram
        prefix = n_gram[:-1]  # The first (n-1) characters as the prefix
        char = n_gram[-1]     # The nth character

        # Update frequency tables for each k from 1 to n
        for k in range(1, n + 1):
            prefix_k = n_gram[:k-1]  # Prefix of length k-1
            char_k = n_gram[k-1]     # k-th character
            frequency_tables[k-1][prefix_k][char_k] += 1  # Increment the count

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
    length = len(sequence)

    # Check if the sequence length is valid
    if length == 0 or length > len(tables):
        return 0.0  # Invalid sequence; no probability can be calculated

    # Get the corresponding frequency table
    freq_table = tables[length - 1]
    
    # Get the frequency of the sequence (as a prefix)
    frequency_of_sequence = sum(freq_table[sequence].values())
    
    # Get the frequency of the specific character following the sequence
    frequency_of_char = freq_table[sequence][char]

    # Calculate the probability
    if frequency_of_sequence > 0:
        probability = frequency_of_char / frequency_of_sequence
    else:
        probability = 0.0  # If no occurrences of the sequence, probability is 0

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
    max_probability = 0.0
    predicted_char = None

    # Iterate through each character in the vocabulary
    for char in vocabulary:
        # Calculate the probability of char occurring after sequence
        probability = calculate_probability(sequence, char, tables)

        # Update the predicted character if this probability is higher
        if probability > max_probability:
            max_probability = probability
            predicted_char = char

    return predicted_char if predicted_char is not None else ''
