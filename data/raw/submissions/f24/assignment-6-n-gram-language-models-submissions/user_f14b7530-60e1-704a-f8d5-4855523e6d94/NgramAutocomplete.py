import utilities as ut;

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    freq_table_list = [] #make table for each length
    for i in range(n+1): # making n frequency tables
        freq_table = {}
        for j in range(len(document)-i): #for each table, find all cases of the specific ngram
            ngram = document[j:(j+i+1)] # for first iteration ngram will be one char, for nth iteration it will be n chars
            if len(ngram) == 1:
                newChar = ngram
                prevChars = ""
            else:
                newChar = ngram[-1]
                prevChars = ngram[:-1]
            if newChar not in freq_table:
                freq_table[newChar] = {}
            if prevChars in freq_table[newChar]:
                freq_table[newChar][prevChars] += 1
            else:
                freq_table[newChar][prevChars] = 1
        freq_table_list.append(freq_table)
    return freq_table_list


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
    if len(sequence)>= len(tables):
        return 0.0
    if len(sequence) == 0:
        nested_dict = tables[0]
        sumTotal = 0
        for outer_key, inner_dict in nested_dict.items():
            sumTotal += sum(inner_dict.values())
        return tables[0][char][""]/sumTotal
    sequence_last = sequence[-1]
    sequence_all_but_last = sequence[:-1]
    if sequence_all_but_last in tables[len(sequence)-1][sequence_last]:
        sequence_count = tables[len(sequence)-1][sequence_last][sequence[:-1]]
    else:
        return 0.0
    if sequence in tables[len(sequence)][char]:
        sequence_with_char_count = tables[len(sequence)][char][sequence]
    else:
        return 0.0
    return sequence_with_char_count/sequence_count


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
    best = ''
    maxVal  = 0
    for i in vocabulary:
        prob = calculate_probability(sequence,i,tables)
        if prob > maxVal :
            maxVal = prob
            best = i
    return best
