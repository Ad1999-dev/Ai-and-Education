from collections import defaultdict

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing
    character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables, where each table is represented as a dictionary.
    """
    frequency_tables = []
    
    # Generate n-grams and their frequency tables
    for i in range(1, n + 1):
        freq_table = defaultdict(lambda: defaultdict(int))
        
        # Create n-grams from the document
        for j in range(len(document) - i):
            ngram = document[j:j + i]
            next_char = document[j + i] if (j + i) < len(document) else None
            
            # Only add to the frequency table if the next_char is not None
            if next_char is not None:
                freq_table[ngram][next_char] += 1
        
        # Convert the defaultdict to a regular dict for cleaner output
        frequency_tables.append(dict(freq_table))
        
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
    # Length of the sequence
    n = len(tables)  # Get the number of frequency tables
    sequence_length = len(sequence)

    # Determine the appropriate frequency table based on the length of the sequence
    if sequence_length == 0:
        return 0.0  # Probability is not defined for an empty sequence

    # Set the appropriate table to use based on the length of the sequence
    if sequence_length > n:
        sequence = sequence[-n:]  # Use only the last n characters
    
    index = sequence_length - 1  # The last character to correspond to its n-gram table
    freq_table = tables[index]    # Select the appropriate frequency table
    
    # Get the total count of characters that follow the sequence
    total_count = sum(freq_table.get(sequence, {}).values())
    
    # Get the count of the specific character following the sequence
    char_count = freq_table.get(sequence, {}).get(char, 0)

    # Calculate the probability
    if total_count > 0:
        probability = char_count / total_count
    else:
        probability = 0.0  # If no characters follow the sequence, probability is zero

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
    max_prob = -1.0  # Initialize maximum probability
    predicted_char = None  # Initialize the predicted character
    
    for char in vocabulary:
        # Calculate the probability of each character following the sequence
        prob = calculate_probability(sequence, char, tables)
        
        # Update the predicted character if we find a higher probability
        if prob > max_prob:
            max_prob = prob
            predicted_char = char
    
    return predicted_char