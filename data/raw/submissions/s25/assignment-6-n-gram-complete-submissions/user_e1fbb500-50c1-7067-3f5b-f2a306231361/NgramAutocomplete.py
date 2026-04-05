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

    for i in range(n):
        if len(tables) == 0:
            tables.append(defaultdict(int))
        else: 
            def make_counter():
                return defaultdict(int)
            tables.append(defaultdict(make_counter))

    for j in range(len(document)):
        current_char = document[j]
        tables[0][current_char] += 1

        for k in range(1, n):
            if j >= k:
                context = document[j-k : j]
                tables[k][context][current_char] += 1

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
    for k in range(min(len(sequence), n - 1), -1, -1): 
        
        if k == 0:
            context = "" 
            numerator = tables[0].get(char, 0)
            denominator = sum(tables[0].values())
        else:
            context = sequence[-k:] 
            if context in tables[k]:
                numerator = tables[k][context].get(char, 0)
                denominator = sum(tables[k][context].values())
            else:
                continue 

        if denominator > 0:
            return numerator / denominator
        elif k > 0:
            continue
        else: 
            return 0
    return 0


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
    if not vocabulary:
        return None
    probabilities = {}
    for char in vocabulary:
        probabilities[char] = calculate_probability(sequence, char, tables)
    if not probabilities:
        return sorted(list(vocabulary))[0]
    best_char = max(probabilities, key=probabilities.get)
    return best_char
