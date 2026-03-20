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
    freq_tables = []
    for i in range(1, n+1):
        ft = {}
        for j in range(0, len(document), i):
            slice = document[j:j+i]
            ft[slice] = ft.get(slice, 0) + 1
        freq_tables.append(ft)
    return freq_tables


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
    if char is None:
        return 0
    
    sequence = sequence + char
    n = len(tables)
    size_C = sum(tables[0].values())
    freq_x0 = tables[0].get(sequence[0], 0)
    
    if freq_x0 == 0 or size_C == 0:
        return 0  
    
    if n == 1:
        return _unigram(sequence, tables, size_C)
    
    p = freq_x0 / size_C
    
    for i in range(1, n-1):
        numerator = tables[i].get(sequence[0:i+1], 0)
        denominator = tables[i-1].get(sequence[0:i], 0)
        if numerator > 0 and denominator > 0:
            p *= numerator / denominator
        else:
            return 0
        
    for i in range(n-1, len(sequence)):
        numerator = tables[n-1].get(sequence[i-(n-1):i+1], 0)
        denominator = tables[n-2].get(sequence[i-(n-1):i], 0)
        if numerator > 0 and denominator > 0:
            p *= numerator / denominator
        else:    
            return 0
        
    return p

def _unigram(full_sequence, tables, size_C):
    """
    Helper for calculate_probability that uses a unigram model.
    """
    p = 1
    for i in range(0, len(full_sequence)):
        p *= tables[0].get(full_sequence[i], 0) / size_C
    return p
    

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
    best_char = None
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char
    if best_char is None:
        return random.sample(vocabulary, 1)[0]
    return best_char
