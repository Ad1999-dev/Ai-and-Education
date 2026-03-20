from utilities import read_file, print_table
import functools
import operator

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    count = 0
    ret = [{} for _ in range(n+1)] 
    # content = read_file(document)
    while count < n+1:
        for i in range(len(document) - count): #Changed all content to document
            ret[count][document[i : i + count + 1]] = ret[count].get(document[i : i + count + 1], 0) + 1
        count += 1
    return ret

def calculate_probability(sequence, char, tables): #('aa', 'a', from above) 
    # (a) * (f(aa)/f(a)) * (f(aaa)/f(aa)) * (f(aaac) / f(aaa)) # probability of calculating the sequence itself 
    # P(sequence + char) / P(sequence) = f(sequence + char) / f(sequence)
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    str = sequence + char
    arrOfRatios = []
    if(sequence[0] in tables[0]):
        arrOfRatios.append(tables[0][sequence[0]] / functools.reduce(operator.add, tables[0].values()))
    else:
        return 0
    for i in range(1, len(str)):
        if (str[0:i] in tables[i - 1] and str[0:i+1] in tables[i] and tables[i - 1][str[0:i]] != 0):
            arrOfRatios.append(tables[i][str[0:i+1]] / tables[i - 1][str[0:i]])
        else:
            arrOfRatios.append(0)
    return functools.reduce(operator.mul, arrOfRatios)

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
    highestProbability = 0
    highestProbChar = ''
    for char in vocabulary:
        freq = calculate_probability(sequence, char, tables)
        if (freq > highestProbability):
            highestProbability = freq
            highestProbChar = char
    return highestProbChar
