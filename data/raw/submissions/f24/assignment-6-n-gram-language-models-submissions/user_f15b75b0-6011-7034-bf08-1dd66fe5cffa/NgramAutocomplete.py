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
        table = {}
        for j in range(len(document) - i - 1):
            sequence = document[j:j + i + 1]
            next_char = document[j + i + 1]
            if sequence not in table:
                table[sequence] = {}
            if next_char not in table[sequence]:
                table[sequence][next_char] = 0
            table[sequence][next_char] += 1
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
    n = len(sequence)
    if n > len(tables):
        sequence = sequence[-(len(tables) - 1):]
        n = len(sequence)

    if n == 0 or sequence not in tables[n - 1]:
        return 0.0

    context_count = sum(tables[n - 1][sequence].values())
    full_ngram_count = tables[n - 1][sequence].get(char, 0)

    return full_ngram_count / context_count


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
    maxProbability = 0
    BestpredictedChar = None
    
    for char in vocabulary:
        probability = calculate_probability(sequence, char, tables)
        if probability > maxProbability:
            maxProbability = probability
            BestpredictedChar = char
            
    return BestpredictedChar