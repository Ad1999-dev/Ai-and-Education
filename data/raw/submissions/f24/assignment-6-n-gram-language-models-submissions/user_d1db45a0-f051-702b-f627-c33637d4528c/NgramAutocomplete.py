from utilities import read_file
from collections import Counter
from fractions import Fraction

def create_frequency_table(string, n):
    frequency_table = Counter()
    for i in range(n, len(string) + 1):
        frequency_table.update([string[i - n:i]])
    return frequency_table


def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    return [create_frequency_table(document, i) for i in range(1, n + 1)]

def get_frequency(sequence, tables):
    n = len(sequence) if len(sequence) < len(tables) else len(tables)
    return tables[n - 1][sequence[-n:]]

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
    divisor = tables[0].total() if len(tables) == 1 else get_frequency(sequence, tables[:-1])
    return Fraction(get_frequency(sequence + char, tables), divisor) if divisor else "NaN"


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
    
    return max(vocabulary, key = lambda c: calculate_probability(sequence, c, tables))
