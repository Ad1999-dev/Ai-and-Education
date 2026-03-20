from collections import defaultdict

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
    
    # Create table for unigrams (n=1)
    unigram_table = defaultdict(int)
    for char in document:
        unigram_table[char] += 1
    tables.append(unigram_table)
    
    # Create tables for n-grams where n > 1
    for k in range(2, n+1):
        # Create a nested defaultdict for this n-gram size
        ngram_table = defaultdict(lambda: defaultdict(int))
        
        # Count occurrences of each k-gram in the document
        for i in range(len(document) - k + 1):
            context = document[i:i+k-1]  # The preceding k-1 characters
            next_char = document[i+k-1]  # The k-th character
            ngram_table[context][next_char] += 1
            
        tables.append(ngram_table)
    
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
    n = len(tables)
    
    # If sequence is longer than our model can handle, use only the last n-1 characters
    if len(sequence) >= n:
        sequence = sequence[-(n-1):]
    
    context_length = len(sequence)
    
    # For context_length = 0 (unigram model)
    if context_length == 0:
        char_count = tables[0][char]
        total_chars = sum(tables[0].values())
        return char_count / total_chars if total_chars > 0 else 0
    
    # For context_length > 0 (higher-order n-grams)
    # Use the appropriate table based on context length
    table_index = context_length
    if table_index >= n:
        table_index = n - 1
        sequence = sequence[-(n-1):]
    
    # Get counts from the appropriate table
    context = sequence
    next_char = char
    
    # Get count of context+char
    if context in tables[table_index] and next_char in tables[table_index][context]:
        numerator = tables[table_index][context][next_char]
    else:
        numerator = 0
    
    # Get total count of context
    denominator = sum(tables[table_index][context].values()) if context in tables[table_index] else 0
    
    # If denominator is 0, back off to a shorter context
    if denominator == 0:
        if context_length > 1:
            # Back off to a shorter context
            return calculate_probability(sequence[1:], char, tables)
        else:
            # Back off to unigram model
            return calculate_probability("", char, tables)
    
    return numerator / denominator


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
    probabilities = {}
    for char in vocabulary:
        probabilities[char] = calculate_probability(sequence, char, tables)
    
    # Find the character with highest probability
    if not probabilities:
        return None  # No valid predictions
    
    max_prob = -1
    max_char = None
    
    for char, prob in probabilities.items():
        if prob > max_prob:
            max_prob = prob
            max_char = char
    
    return max_char
