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

    frequency_tables = []
    for i in range(1, n + 1):
        freq_table = defaultdict(lambda: defaultdict(int))
        doc_length = len(document)
        
        for j in range(doc_length - i + 1):
            ngram = document[j:j+i]
            
            if i > 1:
                context = ngram[:-1]
                char = ngram[-1]
                freq_table[context][char] += 1
            
            else:
                freq_table[None][ngram] += 1
        
        frequency_tables.append(dict(freq_table))
    
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

    n = len(tables)
    seq_length = len(sequence)

    if seq_length == 0 or seq_length > n:
        return 0

    freq_table = tables[seq_length-1]
    context = sequence

    if context not in freq_table:
        return 0

    total_context_count = sum(freq_table[context].values())
    count_sequence_char = freq_table[context].get(char, 0)

    if total_context_count == 0:
        return 0
    
    probability = count_sequence_char / total_context_count
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
    max_prob = -1 
    predicted_char = None

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        
        if prob > max_prob:
            max_prob = prob
            predicted_char = char
    
    return predicted_char if predicted_char else 'a'
