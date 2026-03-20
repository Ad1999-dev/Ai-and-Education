def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
   
    document = document.lower().replace('\n', '').replace(' ', '')

    # Initialize n empty frequency tables
    freq_tables = [{} for _ in range(n)]
    
    # Iterate through the document to build the frequency tables
    for i in range(len(document)):
        # For each position, we'll update n different n-grams (from length 1 to n)
        for j in range(1, n + 1):
            # Only process if we have enough characters for this n-gram
            if i - j + 1 >= 0:
                # Extract the current n-gram
                ngram = document[i-j+1:i+1]
                
                # Get the current table for this n-gram length
                current_table = freq_tables[j-1]
                
                # Update the frequency table
                if ngram in current_table:
                    current_table[ngram] += 1
                else:
                    current_table[ngram] = 1
    
    return freq_tables

def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    Parameters:
        - sequence: The sequence of characters whose probability we want to compute.
        - tables: The list of frequency tables created by create_frequency_tables(), this will be of size n.
        - char: The character whose probability of occurrence after the sequence is to be calculated.

    Returns:
        - Returns a probability value for the sequence.
    """

    n = len(tables)
    
    # Handle the case where sequence is longer than our n-gram model
    if len(sequence) >= n:
        # Use only the most recent n-1 characters as context
        sequence = sequence[-(n-1):] if n > 1 else ""
    
    # Calculate the probability based on the n-gram model
    sequence_plus_char = sequence + char
    
    # Find the appropriate table based on the context length
    context_length = len(sequence)
    
    # If we have no context or empty sequence
    if context_length == 0:
        # Use unigram probabilities (table at index 0)
        if char in tables[0]:
            # Calculate the total count of all characters
            total_count = sum(tables[0].values())
            return tables[0].get(char, 0) / total_count
        else:
            return 0
    
    # For n-grams where n > 1
    else:
        # The table index we need is the context_length
        table_index = context_length
        
        # Get the frequency of the sequence
        sequence_freq = tables[table_index-1].get(sequence, 0)
        
        # If we've never seen this sequence before
        if sequence_freq == 0:
            return 0
        
        # Get the frequency of the sequence followed by the character
        sequence_char_freq = tables[table_index].get(sequence_plus_char, 0)
        
        # Calculate conditional probability: P(char|sequence) = count(sequence+char) / count(sequence)
        return sequence_char_freq / sequence_freq




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
    best_char = None

    max_context_length = len(tables) - 1
    context = sequence[-max_context_length:] if max_context_length > 0 else ""

    for char in vocabulary:
        prob = calculate_probability(context, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char

    return best_char
    

from collections import defaultdict

# Assume your create_frequency_tables, calculate_probability, predict_next_char functions are already defined above.

# -----------------------------
# 1. Create Frequency Tables
# -----------------------------
if __name__ == "__main__":
    # 1. Training document
    document = "aababcaccaaacbaabcaa"
    n = 3  # Trigram model
    vocabulary = ['a', 'b', 'c']

    # Create frequency tables
    tables = create_frequency_tables(document, n)

    # -----------------------------
    # 2. Print Frequency Tables
    # -----------------------------
    print("\n=== Frequency Tables ===")
    for i, table in enumerate(tables):
        print(f"\nTable {i+1} ({i+1}-gram):")
        for seq, count in sorted(table.items()):
            print(f"{seq}: {count}")

    # -----------------------------
    # 3. Probability Check
    # -----------------------------
    print("\n=== Probability Check ===")
    context = "aa"
    for char in vocabulary:
        prob = calculate_probability(context, char, tables)
        print(f"P({char} | {context}) = {prob:.4f}")

    # -----------------------------
    # 4. Predict Next Character
    # -----------------------------
    print("\n=== Next Character Prediction ===")
    prediction = predict_next_char(context, tables, vocabulary)
    print(f"Predicted next character after '{context}': {prediction}")

import time

def run_experiment(document_path, test_sequence, n_values, vocabulary):
    print(f"\n📘 Running experiment on: {document_path}")

    with open(document_path, 'r', encoding='utf-8') as f:
        document = f.read().lower().replace('\n', '').replace(' ', '')  # same cleaning as in create_frequency_tables

    for n in n_values:
        print(f"\n--- Testing n = {n} ---")
        start = time.time()
        tables = create_frequency_tables(document, n)
        end = time.time()

        predicted = predict_next_char(test_sequence, tables, vocabulary)
        duration = end - start

        print(f"Build time: {duration:.2f} seconds")
        print(f"Predicted next char after '{test_sequence}': {predicted}")

# ------------------------------------------
# Run experiment on real corpus files
# ------------------------------------------
if __name__ == "__main__":
    vocabulary = list("abcdefghijklmnopqrstuvwxyz ")  # simple lowercase vocab

    test_sequence = "the"  # common English prefix
    n_values = [2, 3, 4, 5]

    # Run experiment on War and Peace
    run_experiment("warandpeace.txt", test_sequence, n_values, vocabulary)

    # Run experiment on Alice
    run_experiment("Alice's Adventures in Wonderland.txt", test_sequence, n_values, vocabulary)

