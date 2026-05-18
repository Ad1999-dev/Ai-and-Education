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
    for i in range(1, n+1):
        table_i = {}
        for x in range(len(document) - i + 1):
            gram = document[x:x+i]
            if gram not in table_i:
                table_i[gram] = 1
            else:
                table_i[gram] += 1
        tables.append(table_i)
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
    n = len(tables)
    s = len(sequence)
    full_sequence = sequence + char

    if s == 0 or n==1:
        if char in tables[0]:
            p1 = tables[0][char]
        else:
            return 0
        p2 = sum(val for val in tables[0].values() if isinstance(val, int))
        return p1 / p2
    elif s + 1 <= n:
        if full_sequence in tables[s]:
            p1 = tables[s][full_sequence]
        else:
            return 0
        p2 = tables[s-1][sequence]
        return p1 / p2
    else:
        i = len(full_sequence)-n # the start index of our truncation
        full_truncated = full_sequence[i:] # truncate to make len(full_truncated) == n
        seq_truncated = sequence[i:]
        if full_truncated in tables[n-1]:
            p1 = tables[n-1][full_truncated]
        else:
            return 0
        p2 = tables[n-2][seq_truncated]
        return p1 / p2


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
    best_char = ''
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob >= max_prob:
            max_prob = prob
            best_char = char
    return best_char
