# import pandas as pd
from collections import Counter

def generate_grams(document, n):
    grams = [document[i:i + n] for i in range(len(document)-(n - 1))]
    return grams

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
    
    for i in range(n):
        grams = generate_grams(document, i + 1)
        
        gram_frequency = Counter(grams)
        
        tables.append(gram_frequency)
    
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
    prob = 1
    # print(f"{len(sequence)} - {n}")
    # print(tables[len(sequence) - 2][sequence])
    if len(sequence) < n:
        denom = tables[len(sequence) - 1][sequence]
        if denom == 0:
            prob = 0
        else:
            prob = tables[len(sequence)][sequence + char] / denom
    else:
        for i in range(len(sequence) - n + 1):
            substring = sequence[i:i+n-1]
            # print(substring)
            denom = tables[n - 2][substring]
            
            
            
            if denom == 0:
                prob = 0
                break
            if i == 0:
                prob *= denom
            prob *= tables[n - 1][substring + char] / denom
            
    # print(prob)
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
    
    n = len(tables)
    
    prob = 0
    pred = ''
    
    for char in vocabulary:
        newProb = calculate_probability(sequence, char, tables)
        
        # print(f"{char} : {newProb}")
        if newProb > prob:
            
            prob = newProb
            pred = char
    return pred
