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
    for j in range(n):
        for i in range(j,len(document)):
            char = document[i]
            prev = ""
            if i-j > 0:
                for x in range(i-j, i):
                    prev = prev + document[x]
            if len(prev) == j:
                if char in tables[j]:
                    if prev in tables[j][char]:
                        tables[j][char][prev] += 1
                    else:
                        tables[j][char].update({prev: 1})
                else:
                    tables[j].update({char: {prev: 1}})
    

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
    denominator = 0
    numerator = 0
    if len(sequence) > len(tables):
        return calculate_probability(sequence[1:], char, tables)
    for d in tables[len(sequence)]:
        if ((sequence in tables[len(sequence)][d]) and d == char):
            numerator = tables[len(sequence)][d][sequence]
        else:
            for x in tables[len(sequence)][d].values():
                denominator += x
        
        
    if (len(sequence) > 0):
        return 0 if (numerator == 0 or denominator == 0) else numerator/denominator * calculate_probability(sequence[0:-1], sequence[-1], tables)
    else:
        return 0 if (numerator == 0 or denominator == 0) else numerator/denominator

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
    probs = {}
    for i in vocabulary:
        probs.update({i: calculate_probability(sequence[1:], i, tables)})
    max = {'a': -1}
    for x in probs:
        if probs[x] > list(max.values())[0]:
            max = {x: probs[x]}
    return list(max.keys())[0]
