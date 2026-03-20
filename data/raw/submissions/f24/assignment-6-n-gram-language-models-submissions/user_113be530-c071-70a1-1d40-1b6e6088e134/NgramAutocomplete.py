from collections import Counter, defaultdict
import random

"""
This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

- **Parameters**:
    - `document`: The text document used to train the model.
    - `n`: The number of value of `n` for the n-gram model.

- **Returns**:
    - Returns a list of n frequency tables.
"""
def create_frequency_tables(document, n):

    frequency_tables = []
    for i in range(1, n+1): # i-gram table
        table = defaultdict(int)
        for j in range(len(document) + 1 - i): # upper limit
            n_gram = tuple(document[j : j + i])
            table[n_gram] += 1
        frequency_tables.append(dict(table))

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
    sequence_length = len(sequence)
    if sequence_length >= len(tables):
        # If the sequence length is too large for available tables, return 0
        return 0
    
    # Create n-grams for sequence and sequence+char
    sequence_tuple = tuple(sequence)
    extended_tuple = tuple(sequence + char)
    
    # Get frequencies; table index corresponds to sequence length - 1
    sequence_frequency = tables[sequence_length-1].get(sequence_tuple, 0)
    extended_frequency = tables[sequence_length].get(extended_tuple, 0)
    
    # If sequence frequency is 0, return 0 to avoid division by zero
    if sequence_frequency == 0:
        return 0

    # Compute conditional probability
    probability = extended_frequency / sequence_frequency
    return probability

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
def predict_next_char(sequence, tables, vocabulary):
    table = tables[len(sequence)] # includes the extra char bc zero indexing
    best_freq = 0
    next_char = random.choice(tuple(vocabulary))[0]
    # vocabulary is a set of length-1 tuples

    # consider sequences 
    for c in vocabulary:
        c = c[0]
        candidate = tuple(sequence + c)
        if table.get(candidate, 0) >= best_freq:
            best_freq = table.get(candidate, 0)
            next_char = c

    return next_char
