def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables = [{} for _ in range(n+1)]
    for k in range(1, n + 2):
        for i in range(k - 1, len(document)):
            k_gram = document[i - k + 1 : i + 1]
            current_table = frequency_tables[k-1]
            current_table[k_gram] = current_table.get(k_gram, 0) + 1
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
    n_gram = sequence + char
    n_gram_length = len(n_gram)
    context_length = len(sequence)

    numerator_count = 0
    if n_gram_length > 0 and (n_gram_length - 1) < len(tables):
        numerator_count = tables[n_gram_length - 1].get(n_gram, 0)
    
    denominator_count = 0
    if context_length > 0 and (context_length - 1) < len(tables):
         denominator_count = tables[context_length - 1].get(sequence, 0)

    if numerator_count == 0 or denominator_count == 0:
        probability = 0.0
    else:
        probability = numerator_count / denominator_count

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
    max_probability = -1.0
    predicted_char = None
    for char in vocabulary:
        probability = calculate_probability(sequence, char, tables)
        if probability > max_probability:
            max_probability = probability
            predicted_char = char
    return predicted_char