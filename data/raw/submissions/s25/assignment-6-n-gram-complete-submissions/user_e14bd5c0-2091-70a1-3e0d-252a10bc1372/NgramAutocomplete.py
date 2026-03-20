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
    for k in range(1, n + 1):
        table = defaultdict(lambda: defaultdict(int))
        for i in range(len(document) - k + 1):
            context = tuple(document[i:i + k - 1])
            next_char = document[i + k -1]
            table[context][next_char] += 1
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
    n = len(tables)
    context = tuple(sequence[-(n - 1):]) if n > 1 else tuple()

    table = tables[n - 1]
    next_chars = table.get(context, {})
    total = sum(next_chars.values())

    if total == 0:
        return 0.0

    return next_chars.get(char, 0) / total


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
    best_char = None
    best_prob = -1.0

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > best_prob:
            best_prob = prob
            best_char = char
    
    return best_char
