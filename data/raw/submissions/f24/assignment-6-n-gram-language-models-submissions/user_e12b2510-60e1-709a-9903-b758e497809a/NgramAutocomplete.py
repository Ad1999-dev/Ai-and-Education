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
    for i in range(1, n + 1):
        if i == 1:
            table = defaultdict(int)
        else:
            table = defaultdict(lambda: defaultdict(int))
        
        for j in range(len(document) - i + 1):
            sequence = document[j:j + i]
            if i == 1:
                table[sequence] += 1
            else:
                prev_sequence = sequence[:-1]
                table[prev_sequence][sequence[-1]] += 1
        
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
    n = len(sequence)
    if n == 0:
        return 0
    
    table = tables[n - 1]
    
    if n == 1:
        total = sum(table.values())
        prob = table.get(char, 0) / total
        return prob
    
    prev_sequence = sequence[-(n - 1):]
    numerator = table[prev_sequence].get(char, 0)
    denominator = sum(table[prev_sequence].values())
    prob = numerator / denominator if denominator != 0 else 0
    return prob

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
    max_prob = -1
    predicted_char = ''
    
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        
        if char != ' ' and prob > max_prob:
            max_prob = prob
            predicted_char = char
    
    return predicted_char
