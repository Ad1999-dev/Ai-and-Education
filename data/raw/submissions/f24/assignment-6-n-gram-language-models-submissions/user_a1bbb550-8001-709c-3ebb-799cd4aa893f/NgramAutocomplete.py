def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = [{} for _ in range(n)]
    
    for i in range(len(document)):
        for j in range(1, n + 1):
            if i + j <= len(document):
                sequence = document[i:i + j]
                if sequence not in tables[j - 1]:
                    tables[j - 1][sequence] = 0
                tables[j - 1][sequence] += 1

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
    sequenceCopy = sequence

    if (len(sequenceCopy) >= len(tables)):
        numToRemove = (len(sequenceCopy) + 1) - len(tables)
        sequenceCopy = sequenceCopy[numToRemove:]

    newSequence = sequenceCopy + char

    if (newSequence in tables[len(newSequence) - 1]):
        sequencePlusCharFreq = tables[len(newSequence) - 1][newSequence]
        sequenceCopyFreq = tables[len(sequenceCopy) - 1][sequenceCopy]
        return sequencePlusCharFreq/sequenceCopyFreq
    else:
        return 0


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
    maxProb = {'char': '', 'prob': -1}
    for char in vocabulary:
        curProb = calculate_probability(sequence, char, tables)

        if (curProb > maxProb['prob']):
            maxProb['char'] = char
            maxProb['prob'] = curProb

    return maxProb['char']
