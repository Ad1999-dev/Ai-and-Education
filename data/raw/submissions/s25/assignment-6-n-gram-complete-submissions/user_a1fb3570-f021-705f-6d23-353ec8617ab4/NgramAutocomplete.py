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

    # Initialize n tables, each as defaultdict of d efaultdict(int)
    for _ in range(n):
        tables.append(defaultdict(lambda: defaultdict(int)))
    
    doc_length = len(document)
    
    for i in range(doc_length):
        for level in range(n):
            if i - level < 0:
                continue  # Not enough context characters before position i
            context = document[i - level:i]  # i characters before current
            char = document[i]
            tables[level][char][context] += 1
    
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

    level = len(sequence)  # Pick the correct n-gram level
    
    if level >= len(tables):
        return 0  # Sequence is too long for the model
    
    table = tables[level]  # Get the table for this n-gram level
    
    # Count of `char` appearing after `sequence`
    char_count = table.get(char, {}).get(sequence, 0)
    
    # Total number of any characters appearing after `sequence`
    context_count = sum(
        table[c].get(sequence, 0) for c in table
    )
    
    if context_count == 0:
        return 0  # Avoid division by zero
    
    return char_count / context_count


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
    best_char = None

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char

    return best_char

    # return 'a'
