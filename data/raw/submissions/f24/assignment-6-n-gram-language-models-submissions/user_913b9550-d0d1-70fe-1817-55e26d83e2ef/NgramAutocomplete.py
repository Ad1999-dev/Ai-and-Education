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
    for count in range(n):
        print(1+count)
        table_n = {}
        for i in range(len(document)-1-count):
            if document[i:i+count+1] not in table_n:
                table_n[document[i:i+count+1]] = 1
            else:
                table_n[document[i:i+count+1]] += 1
        tables.append(table_n.copy())
    print(tables)
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
    #print(sequence)
    n = len(tables)
    #print(n)
    full_seq = sequence+char
    print(full_seq)
    doc_Size = 0
    for char in tables[0]:
        doc_Size += tables[0][char]
    if len(sequence) < n:
        try:
            return tables[len(full_seq)-1][full_seq]/tables[len(full_seq)-2][sequence]
        except KeyError:
            return 0
    curProb = tables[0][full_seq[0]]/doc_Size
    #print (curProb)
    for i in range(len(full_seq)-n+1):
        #print(full_seq[i:i+n])
        #print(full_seq[i:i+n-1])
        try:
            curProb = curProb * tables[n-1][full_seq[i:i+n]] / tables[n-2][full_seq[i:i+n-1]]
        except KeyError:
            return 0
    print(char, curProb)
    return curProb


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
    max_prob = 0
    max_char = ""
    for char in vocabulary:
        charProb = calculate_probability(sequence, char, tables)
        if charProb > max_prob:
            max_char = char
            max_prob = charProb
    return max_char
    
