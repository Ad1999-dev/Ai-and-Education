def create_frequency_tables(document, n):
    freqList = []
    for i in range(n):
        table={}
        for j in range(len(document)-i):
            substr = document[j:j+i+1]
            table[substr] = table.get(substr,0)+1
        freqList.append(table)
    return freqList
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

def calculate_probability(sequence, char, tables):
    n = len(tables)
    context_length = min(len(sequence), n - 1)
    context = sequence[-context_length:] if context_length > 0 else ''

   
    if context_length == 0:
        unigram_table = tables[0]
        total = sum(unigram_table.values()) or 1 
        return unigram_table.get(char, 0) / total

   
    numerator_table = tables[context_length]
    denominator_table = tables[context_length - 1] 
    
    combined = context + char
    if len(combined) != context_length + 1:
        return 0.0
    
    numerator = numerator_table.get(combined, 0)
    denominator = denominator_table.get(context, 0)

    return numerator / denominator if denominator else 0.0
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the sequence.
    """


def predict_next_char(sequence, tables, vocabulary):
    if not vocabulary:
        return " "
    
    max_context = len(tables) - 1
    context = sequence[-max_context:] if sequence else ''
    
    best_char = " "
    best_prob = 0.0
    
    for char in vocabulary:
        prob = calculate_probability(context, char, tables)
        if prob > best_prob:
            best_prob = prob
            best_char = char
    
    return best_char
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
