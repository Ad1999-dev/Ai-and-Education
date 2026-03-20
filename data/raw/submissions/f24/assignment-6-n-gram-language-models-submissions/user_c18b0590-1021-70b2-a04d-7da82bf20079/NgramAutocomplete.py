from collections import defaultdict
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

    tables = [defaultdict(int) for _ in range(n)]
    for i in range(len(document)):
        for j in range(1, n + 1): 
            if i + j <= len(document):  
                gram = document[i:i+j]  
                tables[j-1][gram] += 1  
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

    n = len(sequence) + 1

    if n > len(tables):
        sequence = sequence[-(len(tables) - 1):] 
        n = len(tables)

    gram = sequence + char

    alpha = 1
    numerator = tables[n - 1].get(gram, 0) + alpha
    denominator = tables[n - 2].get(sequence, 0) + alpha * len(tables[0]) if n > 1 else sum(tables[0].values()) + alpha * len(tables[0])

    return numerator / denominator if denominator != 0 else 0


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
        probabilities[char] = calculate_probability(sequence, char, tables)

    total_prob = sum(probabilities.values())
    normalized_probs = {char: prob / total_prob for char, prob in probabilities.items()}

    chars, weights = zip(*normalized_probs.items())
    return random.choices(chars, weights=weights, k=1)[0]