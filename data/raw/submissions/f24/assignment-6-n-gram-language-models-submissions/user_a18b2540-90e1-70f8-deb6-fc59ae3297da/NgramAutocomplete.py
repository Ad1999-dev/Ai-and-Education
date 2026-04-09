import random

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
    for x in range(1, n+1):
        table = {}
        if (x < len(document)):
            for y in range(x, len(document)+1):
                if (x == 1):
                    gram = document[y-1]
                elif (x > 1):
                    gram = document[y-x:y]
                if gram in table.keys():
                    table[gram] += 1
                else:
                    table[gram] = 1
        else:
            table[document] = 1
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
    temp_sequence = sequence
    seq_length = len(temp_sequence)
    if (seq_length >= len(tables)):
        temp_sequence = temp_sequence[seq_length-(seq_length-len(tables)-1):]
        seq_length = len(temp_sequence)
    num_table = tables[seq_length]
    den_table = tables[0]
    denominator = 0
    if temp_sequence+char in num_table.keys():
        numerator = num_table[temp_sequence+char]
    else:
        return 0
    for char in den_table:
        denominator += den_table[char]
    return numerator/denominator


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
    largest = 0
    cur_char = ''
    for char in vocabulary:
        temp_prob = calculate_probability(sequence, char, tables)
        if (temp_prob > largest):
            largest = temp_prob
            cur_char = char
        elif (temp_prob == largest):
            cur_char = random.choice([cur_char, char])
    if (cur_char == ''):
        cur_char = random.choice(vocabulary)
    return cur_char
