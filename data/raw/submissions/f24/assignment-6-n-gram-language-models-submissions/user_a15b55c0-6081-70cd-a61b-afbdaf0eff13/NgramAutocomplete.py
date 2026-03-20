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

    # List to store the frequency tables
    frequency_tables = []


    # Create frequency tables for 1 to n grams
    for k in range(1, n + 1):
        freq_table = defaultdict(lambda: defaultdict(int))
        
        # Populate the table by iterating through the document
        for i in range(len(document) - k + 1):
            context = document[i:i + k - 1]  # (k-1)-gram context
            next_char = document[i + k - 1]  # k-th character following the context
            
            # Update the frequency table
            freq_table[context][next_char] += 1
        
        # Convert defaultdict to a regular dict for easier use
        frequency_tables.append(freq_table)
    
    return frequency_tables





def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    n = len(tables)

    # Extract the relevant context from the sequence
    context_length = min(len(sequence), n - 1)
    context = sequence[-context_length:] if context_length > 0 else ""

    # Get the appropriate table for the current context length
    table = tables[context_length]

    # If the context is not in the table, return 0
    if context not in table:
        return 0.0

    # Get the total frequency of all characters for the given context
    context_total = sum(table[context].values())

    # Get the frequency of the target character following the context
    char_frequency = table[context].get(char, 0)

    # Calculate and return the conditional probability
    probability = char_frequency / context_total if context_total > 0 else 0.0
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
    # Dictionary to hold the probability of each character
    probabilities = {}

    # Calculate the probability for each character in the vocabulary
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        probabilities[char] = prob

    # Find the character with the maximum probability
    predicted_char = max(probabilities, key=probabilities.get)

    return predicted_char

