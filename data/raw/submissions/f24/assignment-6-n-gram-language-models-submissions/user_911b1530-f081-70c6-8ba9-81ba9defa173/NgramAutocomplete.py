from utilities import read_file

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables = []
    document = read_file(document)
    
    for i in range(1, n+1):
        frequency = {}
        for j in range(len(document)):
            gram = document[j:j+i]
            if len(gram) == i:
                if gram in frequency:
                    frequency[gram] += 1
                else:
                    frequency[gram] = 1
            
        frequency_tables.append(frequency)

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
    c = 0
    freq_char = 0
    for table in tables:
        for gram in table:
            if len(gram) == 1:
                c += table[gram]
            if gram == sequence+char:
                freq_char = table[gram]
    return freq_char/c


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
    max = 0
    next_char = ''
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max:
            max = prob
            next_char = char
    return next_char
