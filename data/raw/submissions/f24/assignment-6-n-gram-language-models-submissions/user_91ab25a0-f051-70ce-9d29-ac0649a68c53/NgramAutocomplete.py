
def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables = [{} for _ in range(n)]

    parsedDocument = document.lower()

    for i in range(len(parsedDocument)):
        for k in range(1, n + 1):
            if i + k <= len(parsedDocument):
                prefix = parsedDocument[i:i + k - 1] 

                if k == 1:
                    char = parsedDocument[i]
                    if char not in frequency_tables[k-1]:
                        frequency_tables[k-1][char] = 0
                    frequency_tables[k-1][char]
                else:
                    if i + k < len(parsedDocument):
                        next_char = parsedDocument[i + k - 1] 

                    if prefix not in frequency_tables[k - 1]:
                        frequency_tables[k - 1][prefix] = {}

                    if next_char not in frequency_tables[k - 1][prefix]:
                        frequency_tables[k - 1][prefix][next_char] = 0
                    frequency_tables[k - 1][prefix][next_char] += 1
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
    probability = 0
    seqLen = len(sequence)
    currTable = tables[seqLen]
    occurSeq = 0
    totalAppearances = 0
    if sequence in currTable:
        occurSeq = currTable[sequence].get(char, 0)
        totalAppearances = sum(currTable[sequence].values())
    if totalAppearances > 0:
        probability = occurSeq / totalAppearances
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
    seqLen = len(sequence)
    currTable = tables[seqLen]
    highestValue = 0
    highestChar = 'a'
    for char in vocabulary:
        currValue = calculate_probability(sequence, char, tables)
        if currValue > highestValue:
                highestValue = currValue
                highestChar = char
    return highestChar
