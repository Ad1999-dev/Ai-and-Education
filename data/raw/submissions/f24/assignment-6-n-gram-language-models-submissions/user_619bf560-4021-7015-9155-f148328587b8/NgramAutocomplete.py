def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    out_tables = []
    for i in range(n):
        cur_table = {}
        start_index = 0
        end_index = start_index + i + 1

        while (end_index <= len(document)):
            sub_str = document[start_index : end_index]
            cur_table[sub_str] = cur_table.setdefault(sub_str, 0) + (1 / len(document)) # By adding 1/len, we are updating the probability of the sequence

            start_index += 1
            end_index += 1

        out_tables.append(cur_table)

    return out_tables


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

    """
        Want to caluclate P(xm | x1, x2, ... xm-1) = f(x1, x2, ..., xm) / f(x1, x2, ..., xm-1)
        All terms will cancel except for the last, so if m <= n, the last term will be f(x1, ..., xm) / f(x1, ..., xm-1)

        If m > n: the last term will be f(x1+(m-n), x2+(m-n), ... xn+(m-n)) / f(x1+(m-n) + x2+(m-n), ..., xn-1+(m-n))
    """

    full_sequence = sequence + char
    n = len(tables)
    m = len(full_sequence)

    if (m == 1):
        return tables[0][char]

    if (m <= n):
        # If m = 2, we want the numerator to be x1 x2, which is table 1, so tables[m-1]
        if (full_sequence not in tables[m-1]):
            return 0
        
        return tables[m - 1][full_sequence] / tables[m - 2][sequence]
    else:
        if (full_sequence[m-n:m] not in tables[n-1] or full_sequence[m-n:m-1] not in tables[n-2]):
            return 0

        return tables[n - 1][full_sequence[m-n:m]] / tables[n - 2][full_sequence[m-n:m-1]]


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
    most_likely_char = ''
    for c in vocabulary:
        prob = calculate_probability(sequence, c, tables)
        if prob > max_prob:
            max_prob = prob
            most_likely_char = c
        
    return most_likely_char
