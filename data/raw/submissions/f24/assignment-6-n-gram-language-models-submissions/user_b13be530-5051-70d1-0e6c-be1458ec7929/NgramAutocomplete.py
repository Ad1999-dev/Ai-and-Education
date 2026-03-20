from collections import defaultdict

def create_frequency_tables(document, n):
    """
    Creates n frequency tables for an n-gram model.

    Parameters:
    - document: The text document used to train the model.
    - n: The size of the n-gram (e.g., 1 for unigram, 2 for bigram).

    Returns:
    - A list of n nested defaultdicts storing frequencies.
    """
    tables = []
    
    for i in range(n):
        table = defaultdict(lambda: defaultdict(int))
        
        for j in range(len(document) - i):
            if i == 0:
                # Unigram: No context
                table[document[j]][()] += 1
            else:
                # Higher-order n-grams
                if j + i < len(document):  # Prevent out of bounds
                    prev_chars = tuple(document[j:j+i])
                    current_char = document[j+i]
                    table[current_char][prev_chars] += 1
        
        tables.append(table)
    
    return tables

def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of a character given a sequence using n-gram tables.

    Parameters:
    - sequence: The preceding sequence of characters.
    - char: The character to calculate the probability for.
    - tables: The list of frequency tables.

    Returns:
    - The probability of `char` given the `sequence`.
    """
    if not sequence:
        # Handle unigram case
        total_count = sum(sum(freq.values()) for freq in tables[0].values())
        return tables[0][char][()] / total_count if total_count > 0 else 0
    
    # Determine context length
    n = len(tables)
    context_length = min(len(sequence), n-1)
    context = tuple(sequence[-context_length:])
    
    # Get frequencies
    table = tables[context_length]
    
    # Calculate denominator (total occurrences of all chars with the same context)
    denominator = sum(table[c][context] for c in table)
    
    if denominator == 0:
        return 0  # If there are no occurrences of any char with this context
    
    return table[char][context] / denominator

def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character given a sequence.

    Parameters:
    - sequence: The input sequence.
    - tables: The list of frequency tables.
    - vocabulary: The set of all possible characters.

    Returns:
    - The predicted next character.
    """
    max_prob = -1
    prediction = None
    
    # Restrict vocabulary to relevant characters
    possible_chars = vocabulary if not sequence else set(tables[min(len(sequence), len(tables) - 1)].keys())
    
    for char in possible_chars:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            prediction = char
    
    return prediction if prediction is not None else list(vocabulary)[0]
