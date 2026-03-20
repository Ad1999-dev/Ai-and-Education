def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, 
    each table capturing character frequencies with increasing conditional 
    dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = []
    
    document_length = len(document)

    for k in range(1, n+1):
        frequency_table = {}
        
        for i in range(document_length - k + 1):
            ngram = document[i:i+k]
            if ngram in frequency_table:
                frequency_table[ngram] += 1
            else:
                frequency_table[ngram] = 1
        
        tables.append(frequency_table)

    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of 
    characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability 
            we want to compute.
        - `tables`: The list of frequency tables created by 
            `create_frequency_tables()`, this will be of size `n`.
        - `char`: The character whose probability of 
            occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    seq_len = len(sequence)
    n = len(tables)
    doc_len = len(tables[0])

    joint_prob = 1
    seq_prob = 1

    if(n>seq_len):
        full = sequence + char
        if(full not in tables[seq_len]):
            return 0
        return tables[seq_len][full]/tables[seq_len-1][sequence]

    for i in range(1,seq_len+1):
        segment = sequence[max(0,i-n):i]        
       
        if(segment not in tables[len(segment)-1]):
            return 0
        seq_prob *= tables[len(segment)-1][segment]
        if(len(segment) != 1):
            seq_prob /= tables[len(segment)-2][segment[:-1]]
        

    last_n = sequence[max(0,seq_len-(n-1)):]
    if(last_n+char not in tables[len(last_n)]):
        return 0
    joint_prob = seq_prob * tables[len(last_n)][last_n+char]/tables[len(last_n)-1][last_n]

    return joint_prob
    

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
    currentChar = ''
    currentProb = 0
    for i in vocabulary:
        char = i
        prob = calculate_probability(sequence, char, tables)
        if(prob > currentProb):
            currentProb = prob
            currentChar = char
        
    return currentChar
