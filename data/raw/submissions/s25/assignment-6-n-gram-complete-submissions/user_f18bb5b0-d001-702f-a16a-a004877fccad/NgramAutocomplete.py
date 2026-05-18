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

    doc_len = len(document)
    for i in range(doc_len):
        for k in range(n):
            current_n = k + 1
            if i - current_n + 1 >= 0:
                sequence = document[i - current_n + 1 : i + 1]
                freq_table = tables[k]
                freq_table[sequence] = freq_table.get(sequence, 0) + 1
                
    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing `char` after `sequence` using the frequency tables.
    P(char | sequence) is approximated using the n-gram model:
    P(char | last_n-1_chars_of_sequence) = count(last_n-1_chars_of_sequence + char) / count(last_n-1_chars_of_sequence)

    - **Parameters**:
        - `sequence`: The sequence of characters preceding the character to predict.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.

    - **Returns**:
        - Returns the conditional probability P(char | sequence).
    """
    n = len(tables)
    
    if n == 1:
        context_sequence = ""
    else:
        context_sequence = sequence[-(n - 1):]

    target_sequence = context_sequence + char

    target_table_index = len(target_sequence) - 1
    context_table_index = len(context_sequence) - 1

    numerator_freq = 0
    if 0 <= target_table_index < n:
        numerator_freq = tables[target_table_index].get(target_sequence, 0)

    denominator_freq = 0
    if context_sequence == "":
        if 0 < n:
             denominator_freq = sum(tables[0].values())
        if 0 < n:
             numerator_freq = tables[0].get(char, 0)
    elif 0 <= context_table_index < n:
        denominator_freq = tables[context_table_index].get(context_sequence, 0)

    if denominator_freq == 0:
        return 0.0
    else:
        return numerator_freq / denominator_freq


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
    max_prob = -1.0
    best_char = None

    if not vocabulary:
        return None

    best_char = sorted(list(vocabulary))[0] 

    for candidate_char in vocabulary:
        prob = calculate_probability(sequence, candidate_char, tables)
        
        if prob > max_prob:
            max_prob = prob
            best_char = candidate_char
            
    return best_char
