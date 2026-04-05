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
    document = document.lower()
    document = document.strip()
    frequency_tables = []
    for k in range(1, n + 1):
        freq_table = defaultdict(lambda: defaultdict(int))
        for i in range(len(document) - k + 1):
            context = document[i:i + k - 1]
            char = document[i + k - 1]
            freq_table[char][context] += 1
        frequency_tables.append(freq_table)
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
    n = len(tables)
    context = sequence[-(n-1):]
    k = len(context) + 1
    if k > n:
        return 0.0
    freq_table = tables[k-1]
    char_count = freq_table[char].get(context, 0)
    context_count = sum(freq_table[ch].get(context, 0) for ch in freq_table)
    if context_count == 0:
        return 0.0
    return char_count / context_count



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
    if best_char is not None:
        return best_char
    else:
        return ""

