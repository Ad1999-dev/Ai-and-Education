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
    frequency_tables = [{} for _ in range(n+1)]
    length = len(document)

    for i in range(length):
        for subNIndex in range(1, n + 2):
            if i + subNIndex <= length:
                # Get the current character (n-gram)
                n_gram = document[i + subNIndex - 1]
                # Determine the preceding context (characters before current n-gram)
                if subNIndex == 1:
                    preceding_context = ""
                else:
                    preceding_context = document[i:i + subNIndex - 1]
                key = n_gram
                # Create new entry in frequency table if this character hasn't been seen before
                if key not in frequency_tables[subNIndex - 1]:
                    frequency_tables[subNIndex - 1][key] = {}
                # Update the count for this character given its preceding context
                if preceding_context in frequency_tables[subNIndex - 1][key]:
                    frequency_tables[subNIndex - 1][key][preceding_context] += 1
                else:
                    frequency_tables[subNIndex - 1][key][preceding_context] = 1

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
    seqLength = len(sequence)
    # Check if the sequence length is valid
    if seqLength > n or seqLength < 0:
        return 0.0
    # Select the appropriate frequency table based on sequence length
    frequency_table = tables[seqLength]

    if seqLength > 0:
        # Get the context from the sequence
        preceding_context = sequence[-(seqLength):]
        key = char
    else:
        # Handle the case where there is no context
        preceding_context = ""
        key = char

    count_key = 0
    # Check if the character exists in the frequency table
    if key not in frequency_table:
        count_key = 0
    else:
        # Get the count of this character following the given context
        count_key = frequency_table[key].get(preceding_context, 0)

    total_count = 0
    # Calculate total count differently based on context length
    if(seqLength == 0):
        # For 0-order model, sum all counts across the table
        for k in frequency_table.keys():
            for num in frequency_table[k].keys():
                total_count += frequency_table[k][num]
    else:
        # For higher-order models, look up the count of the context in the previous table
        prev_table = tables[seqLength - 1]
        prev_key = preceding_context[-1:]
        prev_context = preceding_context[:-1]
        if prev_key in prev_table:
            total_count = prev_table[prev_key].get(prev_context, 0)
    # Avoid division by zero
    if total_count == 0:
        return 0.0
    # Calculate conditional probability
    probability = count_key / total_count
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
    max_probability = 0.0
    predicted_char = ""
    
    # Loop through every character in the vocabulary to calculate its probability
    for char in vocabulary:
        # Calculate probability of char given the sequence
        probability = calculate_probability(sequence, char, tables)
        # Check if probability is the highest found so far
        if probability > max_probability:
            max_probability = probability
            predicted_char = char
    
    return predicted_char



def test_ngram_model():
    """
    Test function to verify the N-gram language model implementation works correctly.
    """
    # 1. Test with a simple document and n=2 (bigram model)
    test_doc = "hello hello world"
    n = 2
    
    print("Testing N-gram model with document:", test_doc)
    print("Using n =", n)
    
    # 2. Create frequency tables
    tables = create_frequency_tables(test_doc, n)
    
    # 3. Print frequency tables for verification
    print("\nFrequency Tables:")
    for i in range(n):
        print(f"Table {i+1} (n={i+1}):")
        for char in sorted(tables[i].keys()):
            for context in sorted(tables[i][char].keys()):
                print(f"  P({char} | {context}) = {tables[i][char][context]}")
    
    # 4. Test calculation of some probabilities
    vocab = set(tables[0].keys())
    print("\nTesting probabilities:")
    
    # Test some specific characters after different contexts
    test_cases = [
        ("h", "l"),
        ("e", "h"),
        ("o", "l"),
        ("w", " "),
        ("d", "l")
    ]
    
    for char, context in test_cases:
        prob = calculate_probability(context, char, tables)
        print(f"P({char} | {context}) = {prob:.4f}")
    
    # 5. Test prediction of next character
    test_sequences = ["h", "he", "hel", "hell", " h", " w"]
    print("\nTesting predictions:")
    
    for seq in test_sequences:
        predicted = predict_next_char(seq, tables, vocab)
        print(f"After '{seq}', predicted next char: '{predicted}'")
    
    # 6. Test a small autocomplete example
    initial = "he"
    k = 5
    current = initial
    
    print(f"\nAutocompleting '{initial}' with k={k}:")
    print(f"Initial: {current}")
    
    for i in range(k):
        next_char = predict_next_char(current[-n:], tables, vocab)
        current += next_char
        print(f"Step {i+1}: {current}")
    
    print("\nTest completed.")


# Uncomment the line below to run the test function when this file is executed directly
if __name__ == "__main__":
     test_ngram_model()