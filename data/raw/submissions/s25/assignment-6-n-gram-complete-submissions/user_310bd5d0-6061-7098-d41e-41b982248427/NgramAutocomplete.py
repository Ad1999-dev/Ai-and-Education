from collections import defaultdict
from utilities import print_table
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
    
    for i in range(n+1):
        table = defaultdict(int)
        
        for j in range(len(document) - i):
            ngram = document[j:j + i + 1]
            
            table[ngram] += 1
        
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
    n = len(sequence) # This could also be len(tables) - 1 but it seems like main.py passes in the updated trailing sequences of length n
    
    sequence_and_char_count = tables[n][sequence + char] # count of the sequence with its predicted character in the document
    
    sequence_count = tables[n-1][sequence] # count of the sequence in the document
    
    return sequence_and_char_count / sequence_count if sequence_count > 0 else 0


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
    probabilities = {}
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        probabilities[char] = prob
    
    return max(probabilities, key=probabilities.get)
