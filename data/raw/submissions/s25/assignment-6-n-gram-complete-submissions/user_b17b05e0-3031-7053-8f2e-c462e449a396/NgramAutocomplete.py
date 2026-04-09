import collections

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = [collections.defaultdict(int) for _ in range(n)]

    L = len(document)

    for k in range(1, n+1):
        for i in range(L - k + 1):
            gram = tuple(document[i:i+k])
            tables[k-1][gram] += 1

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
    max_ctx = len(tables) - 1
    
    ctx = tuple(sequence[-max_ctx:]) if len(sequence) > max_ctx else tuple(sequence)
    k = len(ctx)

    if k >= len(tables):
        return 0.0

    full = ctx + (char,)
    count_full = tables[k].get(full, 0)

    if k == 0:
        total = sum(tables[0].values())
    else:
        total = tables[k-1].get(ctx, 0)

    return (count_full / total) if total > 0 else 0.0


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
    max_prob = -1.0
    best_char = None

    for c in vocabulary:
        p = calculate_probability(sequence, c, tables)
        if p > max_prob:
            max_prob = p
            best_char = c

    if best_char == None:
        most_common_tuple = max(tables[0], key=tables[0].get)
        best_char = most_common_tuple[0]

    return best_char