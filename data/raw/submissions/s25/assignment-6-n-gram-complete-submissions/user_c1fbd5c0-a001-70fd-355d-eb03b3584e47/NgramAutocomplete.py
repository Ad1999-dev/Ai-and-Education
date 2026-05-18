def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables = []

    if (n == 0):
        frequency_table_0 = {'': {}}
        frequency_tables.append(frequency_table_0)
        return frequency_tables
    
    frequency_table_1 = {}
    for char in document:
        if char not in frequency_table_1:
            frequency_table_1[char] = {}
        if '' in frequency_table_1[char]:
            frequency_table_1[char][''] += 1
        else:
            frequency_table_1[char][''] = 1
    frequency_tables.append(frequency_table_1)
    
    for i in range(2, n + 1):
        frequency_table_i = {}
        for j in range(len(document) - i + 1):
            ngram = tuple(document[j:j + i])
            prev_chars = ngram[:-1]
            char = ngram[-1]
            
            if char not in frequency_table_i:
                frequency_table_i[char] = {}
            if prev_chars in frequency_table_i[char]:
                frequency_table_i[char][prev_chars] += 1
            else:
                frequency_table_i[char][prev_chars] = 1
        frequency_tables.append(frequency_table_i)
    
    return frequency_tables

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
    ngram = len(tables)
    full_sequence = list(sequence) + [char]
    probability = 1.0
    
    for i in range(len(full_sequence)):
        start = max(0, i - (ngram -1))
        if (i == 0):
            context = ''
        else:
            context = tuple(full_sequence[start:i])
        target = full_sequence[i]
        relevant_table = tables[len(context)]
        
        if target in relevant_table and context in relevant_table[target]:
            total_count = sum(relevant_table[c].get(context, 0) for c in relevant_table)
            if (total_count):
                prob = relevant_table[target][context]/total_count
            else:
                return 0
            probability *= prob
        else:
            return 0
    
    return probability


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
    highest_prob = 0
    
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if (prob > highest_prob):
            highest_prob = prob
            next_char = char
    
    return next_char
