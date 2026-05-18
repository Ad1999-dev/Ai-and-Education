def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables = [defaultdict(int) for _ in range(n)]
    
    # Process the document to fill in the frequency tables
    for i in range(len(document)):
        # For every valid n-gram size from 1 to n
        for k in range(1, n + 1):
            # Get the n-gram
            if i >= k - 1:  # Ensure we have at least k characters
                ngram = document[i - k + 1:i + 1]
                preceding = document[i - k:i]  # The k-1 preceding characters
                frequency_tables[k - 1][preceding][ngram[-1]] += 1
                
    return frequency_tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a character given a sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters before the character whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the character given the sequence.
    """
    n = len(tables)

    # Determine the length of the sequence to locate the appropriate frequency table
    k = len(sequence)
    
    if k > n:
        raise ValueError("The provided sequence length exceeds the available n-gram tables.")

    # Access the correct frequency table based on the length of the sequence
    frequency_table = tables[k - 1]
    
    # Obtain the total count of occurrences for the provided sequence
    total_context_count = sum(frequency_table[sequence].values()) if sequence in frequency_table else 0
    # Get the count of the specific character following the context
    total_char_count = frequency_table[sequence][char] if char in frequency_table[sequence] else 0
    
    # Calculate the probability: ensure no division by zero
    if total_context_count == 0:
        return 0.0  # No occurrences found for the context

    probability = total_char_count / total_context_count
    
    return probability

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
    # Ensure the current sequence is valid for prediction
    if len(sequence) == 0:
        raise ValueError("The input sequence cannot be empty.")

    # Get the length of the current sequence to determine the table to use
    k = len(sequence)
    
    # Initialize variables to track the maximum probability and the best character
    max_probability = -1.0
    predicted_char = None
    
    # Iterate over every character in the vocabulary
    for char in vocabulary:
        probability = calculate_probability(sequence, tables, char)
        
        # Update the predicted character if the current probability is greater
        if probability > max_probability:
            max_probability = probability
            predicted_char = char

    return predicted_char
