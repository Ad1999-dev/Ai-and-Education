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
    length = len(document)
    for i in range(1,n+1):
        table = defaultdict(lambda: defaultdict(int))
        for j in range(length - i + 1):
                context = document[j : j + i - 1]
                next_char = document[j + i - 1]
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
    max = n - 1 #As long as the sequence can be given the amount of tables we have
    history_len = min(len(sequence), max)
    cut_sequence = sequence[-history_len:]
    if history_len == 0:
        counts = tables[0]
        num = counts.get(char,0)
        den = sum(counts.values())
    else: 
        table = tables[history_len]
        counts = table.get(cut_sequence,{})
        num = counts.get(char,0)
        den = sum(counts.values())
    if den == 0:
        return 0
    return (num / den)
        


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
    prob = float('-inf')
    char = ''
    for c in vocabulary:
        prob_calc = calculate_probability(sequence,c,tables)
        if prob_calc > prob:
            prob = prob_calc
            char = c
    return char
