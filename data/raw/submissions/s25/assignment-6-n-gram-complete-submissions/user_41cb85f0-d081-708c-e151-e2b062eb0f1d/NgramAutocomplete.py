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
    frequency_tables = [defaultdict(int) for _ in range(n)]

    for k in range(1, n + 1):  
        for i in range(len(document) - k + 1):
            ngram = document[i:i + k]
            frequency_tables[k - 1][ngram] += 1 

    return frequency_tables

    #return []



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
    k = len(sequence)

    if k == 0:
        total = sum(tables[0].values())
        return tables[0].get(char, 0) / total if total > 0 else 0.0

    full_ngram = sequence + char

    if k+1 > len(tables): 
        return 0.0
    
    numerator = tables[k].get(full_ngram, 0)
    denominator = tables[k - 1].get(sequence, 0)
    return numerator / denominator if denominator > 0 else 0.0

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
    #return 'a'
    best_char = ''
    best_prob = 0.0

    for char in vocabulary:
        p = calculate_probability(sequence, char, tables)
        if p > best_prob or (p == best_prob and char < best_char):  # lexicographic tie-breaker
            best_char = char
            best_prob = p

    return best_char
