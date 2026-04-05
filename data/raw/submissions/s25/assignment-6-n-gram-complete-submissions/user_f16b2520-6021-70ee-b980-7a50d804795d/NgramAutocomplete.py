def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    #Create list
    frequency_tables = []
    #For each table:
    for i in range(1, n+1):
        frequency_table = {}
        #Run through string
        for index in range(i, len(document) + 1):
            #Update frequency
            substring = document[index-i:index]
            if (substring not in frequency_table):
                frequency_table[substring] = 0
            frequency_table[substring] += 1
        #Append frequency_table object to list
        frequency_tables.append(frequency_table)
    #Print
    #print(f"Frequency Tables: {frequency_tables}")    
    #Return
    return frequency_tables


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
    current_table = len(sequence)

    if current_table == 0:
            corpus_length = sum(tables[0].values())
            return tables[0].get(char, 0) / corpus_length if corpus_length > 0 else 0
    
    if current_table >= len(tables):
        return calculate_probability(sequence[1:], char, tables)
    
    frequency_length = sum(tables[current_table - 1].values())

    if frequency_length == 0:
        return 0

    denominator = tables[current_table - 1].get(sequence, 0)
    numerator = tables[current_table].get(sequence + char, 0)
    if denominator == 0:
        return 0
    return numerator / denominator * calculate_probability(sequence[:-1], sequence[-1], tables)

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
    max_character = None
    max_probability = -1
    for character in vocabulary:
        probability = calculate_probability(sequence, character, tables)
        if (probability > max_probability):
            max_probability = probability
            max_character = character
    return max_character
