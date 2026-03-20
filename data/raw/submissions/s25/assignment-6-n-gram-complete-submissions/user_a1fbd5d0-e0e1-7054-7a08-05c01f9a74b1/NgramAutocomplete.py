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
    
    # Generate n-grams for each possible size (1 to n)
    for i in range(len(document)):
        for j in range(1, n + 1):  # Generate unigrams, bigrams, trigrams, etc.
            if i + j <= len(document):
                ngram = tuple(document[i:i+j])  # Generate the n-gram as a tuple
                frequency_tables[j-1][ngram] += 1  # Increment the frequency of the n-gram

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
    n = len(sequence)
    
    # Handle edge case for empty sequence (if needed)
    if n == 0:
        return 0  # Sequence is empty; return 0 probability
    
    # For unigrams (n=1), we just need the frequency of the character
    if n == 1:
        total_unigrams = sum(tables[0].values())
        return tables[0][(char,)] / total_unigrams if total_unigrams > 0 else 0
    
    # For higher order n-grams (bigram, trigram, etc.), calculate the probability
    numerator = tables[n-1].get(tuple(sequence + [char]), 0)
    denominator = tables[n-2].get(tuple(sequence), 0)
    
    if denominator == 0:
        return 0  # Return 0 if the denominator is 0 (use smoothing if desired)
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
    max_prob = -1
    predicted_char = ''
    
    # Calculate the probability for each character in the vocabulary
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        
        if prob > max_prob:
            max_prob = prob
            predicted_char = char

    return predicted_char
