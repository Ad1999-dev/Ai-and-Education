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
                context = document[i:i + j - 1]
                char = document[i + j - 1]
                
                if j == 1:
                    if char not in tables[0]:
                        tables[0][char] = {}
                    if '' not in tables[0][char]:
                        tables[0][char][''] = 0
                    tables[0][char][''] += 1
                else:
                    if char not in tables[j-1]:
                        tables[j-1][char] = {}
                    if context not in tables[j-1][char]:
                        tables[j-1][char][context] = 0
                    tables[j-1][char][context] += 1
    
    for i in range(n):
        context_totals = {}
        for char in tables[i]:
            for context in tables[i][char]:
                if context not in context_totals:
                    context_totals[context] = 0
                context_totals[context] += tables[i][char][context]
        
        for char in tables[i]:
            for context in tables[i][char]:
                if context_totals[context] > 0:
                    tables[i][char][context] /= context_totals[context]
    
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
    index_table = len(sequence)
    if index_table >= len(tables):
        index_table = len(tables) - 1
        sequence = sequence[-(index_table):]
    
    if index_table == 0:
        sequence = ''
    
    if char in tables[index_table] and sequence in tables[index_table][char]:
        return tables[index_table][char][sequence]
    
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
    max_prob = -1
    predict_char = None
    
    n = len(tables)
    if len(sequence) > n - 1:
        sequence = sequence[-(n-1):]
    
    for c in vocabulary:
        prob = calculate_probability(sequence, c, tables)
        if prob > max_prob:
            max_prob = prob
            predict_char = c
    
    if predict_char is None:
        max_freq = -1
        for char in tables[0]:
            if '' in tables[0][char] and tables[0][char][''] > max_freq:
                max_freq = tables[0][char]['']
                predict_char = char
        
        if predict_char is None:
            if vocabulary:
                predict_char = next(iter(vocabulary))
            else:
                predict_char = 'a'
    
    return predict_char
