import math

def increment_dict_value(table, string):
    if string in table:
        table[string] += 1
    else:
        table[string] = 1

def create_frequency_tables(document: str, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    count_tables = []
    for i in range(1, n+1):
        table = {}
        for j in range(len(document) - i + 1):
            increment_dict_value(table, document[j: j+i])
        count_tables.append(table)
    return count_tables

# 4-gram
# P('a' | 'bcd') = f('bcda') / f('bcd')
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
    max_grams = len(tables)
    n = min(len(sequence) + 1, max_grams)

    if n == 1:
        return tables[0][char] / sum([tables[0][char] for char in tables[0]])

    # tables[i-1] has the frequency table for i-grams
    sequence = sequence[-n+1:]
    combined_sequence = sequence + "" + char
    if combined_sequence in tables[n-1]:
        numerator = tables[n-1][combined_sequence]
        # print("numerator not 0")
    else:
        numerator = 0

    if sequence in tables[n-2]:
        denominator = tables[n-2][sequence]
        # print("denominator not 0")
    else:
        denominator = 0
    # print(f"P({char}|{sequence}) =", numerator / denominator if denominator != 0 else 0)
    return numerator / denominator if denominator != 0 else 0


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
    # print("vocab:", vocabulary)
    probabilities = []
    for char in vocabulary:
        probabilities.append((char, calculate_probability(sequence, char, tables)))
    # print(probabilities)
    max_prob = -math.inf
    best_char = ''
    for char, prob in probabilities:
        if max_prob < prob:
            max_prob = prob
            best_char = char

    return best_char