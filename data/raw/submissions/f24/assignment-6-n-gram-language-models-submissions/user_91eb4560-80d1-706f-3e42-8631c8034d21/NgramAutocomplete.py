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
    tables = []
    
    document = document.replace('\n', ' ')
    
    for i in range(1, n + 1):
        table = defaultdict(int)
        for j in range(len(document) - i + 1):
            ngram = document[j:j+i]
            table[ngram] += 1
        
        tables.append(dict(table))
    
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

    n = len(sequence)
    if n >= len(tables):
        sequence = sequence[-len(tables)+1:]
        n = len(sequence)
    current_table = tables[n]
    ngram = sequence + char
    ngram_freq = current_table.get(ngram, 0)
    sequence_freq = tables[n - 1].get(sequence, 0) if n > 0 else sum(tables[0].values())
    if sequence_freq == 0: 
        return 1 / (sum(tables[0].values()) + len(tables))
    return (ngram_freq / sequence_freq)

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
        if prob > max_prob:
            max_prob = prob
            best_char = char
    return best_char
