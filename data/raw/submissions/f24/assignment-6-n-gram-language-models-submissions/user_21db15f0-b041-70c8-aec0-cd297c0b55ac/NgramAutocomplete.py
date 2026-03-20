from utilities import read_file, print_table
def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = [dict[str, int]()] * n
    print("tables is initialized as ", tables)
    cur_n = 0
    while cur_n < n:
        for i, c in enumerate(document):
            if c == '':
                continue
            start_index = i - (cur_n + 1)
            if (start_index < 0):
                start_index = 0
            token = document[start_index:i]
            if token not in tables[cur_n]:
                tables[cur_n][token] = 0
            tables[cur_n][token] += 1
        cur_n += 1

    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    sequence = sequence+char
    if len(sequence) > len(tables):
        return 0
    n = len(sequence)
    table = tables[n-1]
    if sequence not in table:
        return 0
    total_tokens = 0
    for value in table.values():
        total_tokens += value
    probability = table[sequence] / total_tokens
    print("probability of sequence " + sequence + ": " + str(probability))
    return probability


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
    # find x that maximizes P(x | sequence)
    if len(sequence) + 1 > len(tables):
        sequence = sequence[-len(tables)+1:]
    print("sequence is " + sequence)
    best_char = ''
    max_probability = 0
    for char in vocabulary:
        if len(char) == 0:
            continue
        new_probability = calculate_probability(sequence, char, tables)
        if new_probability > max_probability:
            max_probability = new_probability
            best_char = char
            print('best_char is now ' + best_char)
    if (best_char == ''):
        if len(tables) <= 1:
            print("Warning: Your table is only 1 gram long. You need at least bigram tables in order to get a meaningful prediction.")
        print("No likely sequence found (the initial sequence likely isn't in the text)")
        print("Returning empty string.")
    else:
        print("Best prediction is " + best_char)
    return best_char
