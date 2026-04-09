def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    # Initialize a list to hold n frequency tables
    frequency_tables = []
    
    # Create the tables for 1 to n grams
    for i in range(1, n+1):
        # Initialize dictionary for current n-gram level
        table = {}
        
        # Loop through the document to count n-grams
        for j in range(len(document) - i + 1):
            # Get the current n-gram
            if i == 1:
                # For unigrams, we just need the character
                ngram = document[j]
                prev_chars = ""
            else:
                # For bigrams and higher, we need the character and its context
                ngram = document[j:j+i]
                prev_chars = ngram[:-1]  # All but the last character
                ngram = ngram[-1]        # The last character
            
            # Initialize the character entry if not present
            if ngram not in table:
                table[ngram] = {}
            
            # Initialize the prev_chars entry if not present
            if prev_chars not in table[ngram]:
                table[ngram][prev_chars] = 0
            
            # Increment the count
            table[ngram][prev_chars] += 1
        
        # Add the table to our list of frequency tables
        frequency_tables.append(table)
    
    return frequency_tables


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
    
    # If sequence is shorter than n-1, we only use the available context
    context_size = min(n-1, len(sequence))
    
    # Get the context (last context_size characters of the sequence)
    context = sequence[-context_size:] if context_size > 0 else ""
    
    # Calculate the probability based on the appropriate n-gram model
    if context_size < n-1:
        # If our context is shorter than n-1, use a lower order n-gram model
        table_index = context_size
        prev_chars = context
    else:
        # Use the full n-gram model
        table_index = n - 1
        prev_chars = context
        
    # Get the appropriate frequency table
    table = tables[table_index]
    
    # Check if the character and context exist in our table
    if char in table and prev_chars in table[char]:
        # The numerator: frequency of the sequence followed by char
        numerator = table[char][prev_chars]
        
        # The denominator: frequency of the sequence
        denominator = 0
        for c in table:
            if prev_chars in table[c]:
                denominator += table[c][prev_chars]
        
        # Calculate probability
        if denominator > 0:
            return numerator / denominator
    
    # If the character or context doesn't exist in our table, return 0
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
    next_char = None
    
    # For each character in the vocabulary, calculate its probability
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        
        # Update the max probability and next character if a higher probability is found
        if prob > max_prob:
            max_prob = prob
            next_char = char
    
    # Return the character with the highest probability
    # If all probabilities are 0, return a default character (first in vocabulary)
    return next_char if next_char is not None else list(vocabulary)[0]
