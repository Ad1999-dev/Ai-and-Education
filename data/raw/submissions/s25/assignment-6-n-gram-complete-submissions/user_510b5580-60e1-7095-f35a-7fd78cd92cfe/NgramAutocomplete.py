def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = {}
    for i in range(0, n):
        table_n = {}

        for j, char in enumerate(document):
            
            if (j < len(document)-(i)):
                k = document[j:j+(i+1)]

                if k in table_n:
                    table_n[k] += 1
                        
                else:
                    table_n[k] = 1
        tables[i] = table_n
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

    sequence = sequence+char
    product = 1

    size_c = 0
    for i in tables[0]:
        size_c+= tables[0][i]

    for i, char in enumerate(sequence):
        context_size = min(i, n-1)
        context = sequence[i-context_size:i+1]
        
        if context in tables[context_size]:
            numer = tables[context_size][context]
        else:
            numer = 0
        
        if context_size == 0:
            denom = size_c
        else:
            if sequence[i-context_size:i] in tables[context_size-1]:
                denom = tables[context_size-1][sequence[i-context_size:i]]
            else:
                denom = 1
        
        product *= numer/denom
    
    return product


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
    next_char = ''
    next_char_p = 0

    for i in vocabulary:
        char_p = calculate_probability(sequence, i, tables)

        if char_p > next_char_p:
            next_char = i
            next_char_p = char_p

    return next_char
