from collections import defaultdict

def create_frequency_tables(document, n):
    tables = []
    for i in range(n):
        tables.append(defaultdict(lambda: defaultdict(int)))

    for i in range(len(document) - n + 1):
        for j in range(1, n + 1):  # j should be from 1 to n
            current_char = document[i + j - 1]
            prefix = document[i:i + j - 1]  # Prefix length is j-1
            tables[j - 1][current_char][prefix] += 1

    return tables

def calculate_probability(sequence, char, tables):
    n = len(tables)
    seq_length = len(sequence)

    # Check if the length of the sequence is less than n-1
    if seq_length < 1:  # Ensure there's at least one character
        return 0

    context = sequence[-(n - 1):]
    context_length = len(context)
    
    if context_length == 0:
        return 0  # If there is no context
    
    if context_length >= n: 
        context_length = n - 1  # Ensure we access valid index

    # f_seq_char might cause IndexError if context_length exceeds bounds
    f_seq_char = tables[context_length].get(char, {}).get(context, 0)
    f_seq = sum(tables[context_length][c][context] for c in tables[context_length])

    if f_seq == 0:
        return 0  # Avoid division by zero

    return f_seq_char / f_seq

def predict_next_char(sequence, tables, vocabulary):
    probabilities = {}
    
    last_char = sequence[-1] if sequence else None  # Get the last character

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)

        # Ensure we don't immediately repeat the last character
        #if char == last_char:
        #    prob = 0
            
        probabilities[char] = prob

    # Filter out characters with zero probability
    valid_predictions = {char: prob for char, prob in probabilities.items() if prob > 0}

    # Handle case when there are no valid predictions
    if not valid_predictions:
        return ' '  # Provide a fallback character

    # Select the character with the highest probability from valid predictions
    next_char = max(valid_predictions, key=valid_predictions.get)

    return next_char if valid_predictions[next_char] > 0 else ' '
