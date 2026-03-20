from collections import Counter

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    # each table - a Counter object 'ab' -> count of string 'ab'
    # for each new character at index i, increment the frequencies of substrings of lengths 1-n (so starting at index i-n+1 to index i)
    tables = [Counter() for i in range(n)]
    for i in range(len(document)):
        for substrLen in range(1, min(n, i)+1):
            substring = document[i-substrLen+1:i+1]
            tables[substrLen-1][substring] += 1
    return tables

def seq_freq(sequence, tables):
    if len(sequence) <= len(tables):
        return tables[len(sequence)-1][sequence]
    else:           # ensure not trying to find sequence that is too long
        n = len(tables)
        return tables[-1][sequence[-n:]]


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
    sf = seq_freq(sequence[-n+1:], tables)
    if (sf == 0):
        return 0
    return seq_freq(sequence[-n+1:]+char, tables)/sf


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
    maxprob = 0
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob >= maxprob:
            maxprob = prob
            maxprobchar = char
    return maxprobchar
        
