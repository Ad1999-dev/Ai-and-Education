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
    frequency_tables = [defaultdict(int) for _ in range(n)]
    
    # Iterate through the document to create n-gram counts
    for i in range(len(document)):
        for j in range(1, n + 1):  # 1 to n for n-grams
            if i + j <= len(document):
                n_gram = document[i:i + j]
                frequency_tables[j - 1][n_gram] += 1
    print(frequency_tables)
                
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
    t = len(sequence)
    
    # Get the correct frequency table based on the length of the sequence
    if t > 0:
        n = min(len(tables), t + 1)
        table = tables[n - 1]  # Get the appropriate n-gram table

        # Prepare context which is the last n-1 characters from the sequence
        context = sequence[-(n-1):]  # Take last (n-1) characters as context
        
        # Compute joint probability P(sequence, next_char)
        joint_freq = table[context + char] if (context + char) in table else 0
        
        # Compute P(sequence)
        context_freq = tables[n-2][context] if context in tables[n-2] else 0
        
        # Probability P(next_char | sequence)
        if context_freq > 0:
            return joint_freq / context_freq
        else:
            return 0.0  # No valid context means we assume zero probability
            
    #else:
        # If sequence is empty, we might predict uniformly over the vocabulary
        #return 1 / len(vocabulary)  # Uniform distribution


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
    predicted_char = None
    
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            predicted_char = char
            
    return predicted_char
