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
    table = [defaultdict(int) for _ in range(n)]
    length = len(document)

    for i in range(1,n+1):
        for j in range(length-i + 1):
            ngram = tuple(document[j:j+i])
            table[i-1][ngram] += 1  
    return table


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

    n = len(sequence) + 1
    if n>len(tables):
        return 0.0

    ngram_with_char = tuple(sequence) + (char,)
    sequence_count = tables[n-2][tuple(sequence)]
    ngram_with_char_count = tables[n-1].get(ngram_with_char,0)

    if sequence_count == 0:
        return 0.0

    probability = ngram_with_char_count / sequence_count
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
    best_char = None

    for char in vocabulary:
        probability = calculate_probability(sequence, char, tables)
        if probability > max_prob:
            max_prob = probability
            best_char = char
    return best_char if best_char else 'a'
