 

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    freq_list = []

    for x in range(1, n + 1):
        freq = {}
        for i in range(len(document) - x + 1):
            ngram = document[i:i+x]
            if ngram in freq:
                freq[ngram] += 1
            else:
                freq[ngram] = 1
        freq_list.append(freq)

    return freq_list
    




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
    
    
    if len(sequence + char) - 1 >= len(tables) or len(sequence + char)  - 2 >= len(tables):
        return 0.0
    
    prob_sequence = tables[len(sequence) - 1].get(sequence, 0)
    prob_char = tables[len(sequence + char) - 1].get(sequence + char, 0)
    if prob_sequence == 0:
        return 0.0

    return prob_char / prob_sequence


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
    best_char = None

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char


    return best_char or ""



