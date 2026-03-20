from utilities import read_file, print_table

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = [{} for _ in range(n)]
    
    # for unigrams
    for char in document:
        if char in tables[0]:
            tables[0][char] += 1
        else:
            tables[0][char] = 1
    
    # for n-grams
    for i in range(1, n): # index of tables
        for j in range(len(document) - i):
            sequence = document[j:j+i+1]
            # Use the sequence as key
            if sequence in tables[i]:
                tables[i][sequence] += 1
            else:
                tables[i][sequence] = 1
    
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


    max_n = len(tables)

    # Truncate to the maximum context length supported by tables
    if len(sequence) > max_n - 1:
        sequence = sequence[-(max_n - 1):]

    n = len(sequence)

    if n == 0:
        # Base case: use unigram probability
        total = sum(tables[0].values())
        return tables[0].get(char, 0) / total if total > 0 else 0.0

    seq = sequence[-n:]  # last n characters as context
    seq_plus_char = seq + char

    seq_count = tables[n - 1].get(seq, 0)
    seq_plus_count = tables[n].get(seq_plus_char, 0)

    if seq_count > 0:
        return seq_plus_count / seq_count
    else:
        # use shorter sequence if the original sequence is not found
        return calculate_probability(sequence[1:], char, tables)



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
    n = len(tables)
    if len(sequence) > n - 1:
        sequence = sequence[-(n-1):]
    
    # find char with highest prob
    max_prob = -1
    best_char = None
    
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char
    
    return best_char
