from collections import defaultdict

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, 
    each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model (string).
        - `n`: The number of value of `n` for the n-gram model (integer).

    - **Returns**:
        - Returns a list of `n` frequency tables (list of dictionaries).
    """
    
    # Initialize frequency tables
    tables = [defaultdict(int) for _ in range(n)]
    length = len(document)

    # Loop through the document to create frequency counts
    for i in range(length):
        # Create n-grams of size 1 to n
        for j in range(n):
            if i + j < length:  # Ensure we don't exceed document length
                n_gram = document[i:i+j+1]  # Get the n-gram
                tables[j][n_gram] += 1  # Increment the count for this n-gram

    # Convert defaultdicts to regular dictionaries for output
    return [dict(table) for table in tables]



def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute (string).
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated (string).

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    n = len(tables)  # Get the number of tables (n)
    L = len(sequence)  # Length of the input sequence

    # If the sequence is empty or if its length exceeds the available tables, return 0 (or an appropriate fallback)
    if L == 0:
        return 0
    if L >= n:
        return 0  

    # Select the frequency table corresponding to the length of the sequence
    freq_table = tables[L]  
    
    # Construct the full and previous sequence
    full_sequence = sequence + char  # sequence concatenated with the character
    prev_sequence = sequence  # Just the initial sequence

    # Frequency of the full sequence
    freq_full_sequence = freq_table.get(full_sequence, 0)
    # Frequency of the previous sequence
    freq_prev_sequence = freq_table.get(prev_sequence, 0)

    # Use Laplace smoothing to avoid division by zero
    if freq_prev_sequence == 0:
        freq_prev_sequence = 1  # Add 1 to avoid the probability being undefined

    # Calculate probability
    probability = freq_full_sequence / freq_prev_sequence

    return probability


def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character based on the given sequence.

    - **Parameters**:
        - `sequence`: The sequence used as input to predict the next character (string).
        - `tables`: The list of frequency tables.
        - `vocabulary`: The set of possible characters (set or list).
    
    - **Returns**:
        - Returns the character with the maximum probability as the predicted next character.
    """
    max_prob = -1  # Initialize maximum probability
    best_char = None  # Initialize the best character

    # Iterate over each character in the vocabulary
    for char in vocabulary:
        # Calculate the probability of the character following the sequence
        prob = calculate_probability(sequence, char, tables)
        
        # Check if this character has the highest probability so far
        if prob > max_prob:
            max_prob = prob
            best_char = char

    return best_char
