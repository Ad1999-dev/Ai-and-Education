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

    for i in range(1, n + 1):  # we want n frequency tables
        table = defaultdict(int)
        for j in range(len(document) - (i - 1)):
            igram = document[j : j + i]
            table[igram] = table[igram] + 1
        tables.append(table)

    return tables # tables[0] gives frequency table for unigram, tables[1] gives frequency table for bigram, etc


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
    seq = sequence + char
    n = len(tables) 
    prob = 1.0

    for i in range(len(seq)):
        context_start = max(0, i - (n - 1))  # determines the context window for our n-gram
        context = seq[context_start:i] # gets the conditional (denominator)
        target = seq[context_start:i+1] # gets the joint probability (numerator)

        k = len(target) - 1  # use the len(target)-gram table at index len(target) - 1

        numerator = tables[k][target]
        if k > 0:
            denominator = tables[k-1][context]
        else: # if we are using a 1-gram model or on first letter of the doc, then we need to calculate the probability as freq of target / total freq of unigrams
            denominator = sum(tables[0].values())

        if denominator == 0: # if this happens, it means numerator is 0 as well
            return 0
        else:
            prob *= (numerator / denominator)

    return prob


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
    prediction = ""
    highest_prob = 0
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if highest_prob < prob:
            highest_prob = prob
            prediction = char
    
    return prediction
