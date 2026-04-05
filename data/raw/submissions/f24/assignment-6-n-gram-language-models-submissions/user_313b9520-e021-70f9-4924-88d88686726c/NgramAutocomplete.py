import re
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

    text = re.sub(r'[^a-zA-Z]', '', document)
    text = text.lower()

    for i in range (1, n+1):
        frequency_table = {}

        for j in range (len(text) - i + 1):
            n_gram = tuple(text[j:j + i])

            if n_gram in frequency_table:
                frequency_table[n_gram] += 1
            else:
                frequency_table[n_gram] = 1
        frequency_tables.append(frequency_table)

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

    if n > len(tables) - 1:
        sequence = sequence[-(len(tables) - 1):]
        n = len(sequence)
    
    target_ngram = tuple(list(sequence) + [char]) 
    condition_ngram = tuple(sequence) 

    if n == 0: 
        target_frequency = tables[0].get((char,), 0)
        total_frequency = sum(tables[0].values())
    else: 
        target_frequency = tables[n].get(target_ngram, 0)
        total_frequency = tables[n - 1].get(condition_ngram, 0)
    
    probability = target_frequency / total_frequency
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

    max_prob = 0 
    best_char = None

    for char in vocabulary:

        prob = calculate_probability(sequence, char, tables)

        if prob > max_prob:
            max_prob = prob
            best_char = char

    return best_char if best_char is not None else ''
