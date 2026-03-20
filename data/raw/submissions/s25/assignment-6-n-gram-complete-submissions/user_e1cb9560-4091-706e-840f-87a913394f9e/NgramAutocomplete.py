# helpers

def createFrequencyTable(document, n):
    table = {}
    for i in range(len(document) - n + 1):
        window = document[i:i + n]
        if window in table:
            table[window] += 1
        else:
            table[window] = 1
    return table

def calculateConditionalProb(target, given, frequencyTable):

    if len(frequencyTable) == 1:
        return frequencyTable[0].get(target, 0)
    
    if given == None:
        return frequencyTable[0][target] / sum(frequencyTable[0].values())
    
    text = given + target
    numerator = frequencyTable[len(given)].get(text, 0)
    denominator = frequencyTable[len(given) - 1].get(given, 0)

    return numerator / denominator if denominator != 0 else 0

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
        tables.append(createFrequencyTable(document, i + 1))
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
    product = 1
    gram = len(tables) - 1 # minus one because you always always create the table the size of gram + 1
    text = sequence + char
    for i in range(len(text)):
        if i == 0:
            product *= calculateConditionalProb(text[i], None, tables)
        else:
            start = max(0, i - gram)
            given = text[start:i]
            product *= calculateConditionalProb(text[i], given, tables)
    return product


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

    # TODO Can return None need to me fix
    nextChar = None
    maxProb = 0
    
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > maxProb:
            maxProb = prob
            nextChar = char
    return nextChar