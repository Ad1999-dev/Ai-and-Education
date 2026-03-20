def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    document = document.lower()
    freq_tables = []
    

    # Create tables for 1-gram to n-gram
    for i in range(1, n+1):
        freq_table = {}
        for j in range(len(document)-i+1):
            string = document[j:j+i]
            freq_table[string] = freq_table.get(string, 0) + 1
        freq_tables.append(freq_table)
    
    return freq_tables
print(create_frequency_tables("aababcaccaaacbaabcaa",3))
# print(create_frequency_tables("hi",5))
def calculate_probability(sequence, char, tables):
    """
    Calculates the joint probability of observing a sequence followed by a character
    using the n-gram model tables and the chain rule of probability.

    - **Parameters**:
        - `sequence`: The sequence of characters preceding the character we want to predict.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`.
        - `char`: The character whose probability we want to calculate.

    - **Returns**:
        - Returns the joint probability P(sequence, char) using the chain rule.
    """
    n = len(tables)
    full_sequence = sequence + char
    
    # Handle first character separately (unigram probability)
    char_i = full_sequence[0]
    char_count = tables[0].get(char_i, 0)
    total_count = sum(tables[0].values())
    
    if total_count == 0 or char_count == 0:
        return 0
            
    joint_prob = char_count / total_count
    
    # Process remaining characters using conditional probability
    for i in range(1, len(full_sequence)):
        context_length = min(i, n-1)
        context = full_sequence[i-context_length:i]
        current_ngram = context + full_sequence[i]
        
        context_table_idx = len(context) - 1
        ngram_table_idx = len(current_ngram) - 1
        
        if context_table_idx >= n or ngram_table_idx >= n or context_table_idx < 0:
            return 0
        
        context_count = tables[context_table_idx].get(context, 0)
        ngram_count = tables[ngram_table_idx].get(current_ngram, 0)
        
        if context_count == 0:
            return 0
        
        conditional_prob = ngram_count / context_count
        joint_prob *= conditional_prob
    
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
    max_prob = 0
    best_char = None

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char

    if best_char is None and vocabulary:
        vocab_list = sorted(list(vocabulary))
        best_char = vocab_list[0]
        
    return best_char
tables = create_frequency_tables("aababcaccaaacbaabcaa",3)
print(predict_next_char("aa", tables, {"a","b","c"}))
