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
    tables = [defaultdict(int) for _ in range(n)]
    
    for i in range(n):
        seq_len = i + 1
        for j in range(len(document) - seq_len + 1):
            sequence = document[j:j + seq_len]
            tables[i][sequence] += 1
            
    return tables

def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    if len(sequence) >= len(tables):
        sequence = sequence[-(len(tables)-1):]  
    
    if len(sequence) == 0:
        total_chars = sum(tables[0].values())
        return tables[0].get(char, 0) / total_chars if total_chars > 0 else 0.0
    
    full_sequence = sequence + char
    n = len(full_sequence)
    
    context_count = tables[n-2][sequence]  
    sequence_count = tables[n-1][full_sequence]  
    
    if context_count == 0:
        if len(sequence) > 1:
            return calculate_probability(sequence[1:], char, tables)
        else:
            return calculate_probability("", char, tables)
    
    return sequence_count / context_count


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
    if len(sequence) >= len(tables):
        sequence = sequence[-(len(tables)-1):]
    
    max_prob = -1
    predicted_char = None
    
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            predicted_char = char
    
    return predicted_char if predicted_char else ""
