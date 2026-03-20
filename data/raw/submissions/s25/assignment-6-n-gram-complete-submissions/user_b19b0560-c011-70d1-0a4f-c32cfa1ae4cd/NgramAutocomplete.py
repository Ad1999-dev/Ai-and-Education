# NgramAutocomplete.py

def create_frequency_tables(document, n):
    if n <= 0:
        return []

    # Create n frequency tables
    tables = [{} for _ in range(n)]

    # Loop through all positions in the document
    for position in range(len(document)):
        for ngram_size in range(1, n + 1):
            if position >= ngram_size - 1:
                # Extract the n-gram
                start_index = position - (ngram_size - 1)
                end_index = position + 1
                ngram = document[start_index:end_index]

                # Update the frequency table
                if ngram in tables[ngram_size - 1]:
                    tables[ngram_size - 1][ngram] += 1
                else:
                    tables[ngram_size - 1][ngram] = 1
    
    return tables


def calculate_probability(sequence, char, tables):
    n = len(tables)
    
    # Trim the sequence to the last (n-1) characters
    if len(sequence) >= n:
        sequence = sequence[-(n - 1):]
    
    # Unigram probability calculation
    if len(sequence) == 0:
        total_count = sum(tables[0].values())
        char_count = tables[0].get(char, 0)
        return char_count / total_count if total_count > 0 else 0
    
    # Determine context length
    context_length = min(len(sequence), n - 1)
    context = sequence[-context_length:]

    # Calculate the extended sequence frequency
    extended_seq = context + char

    context_freq = tables[context_length - 1].get(context, 0)
    extended_freq = tables[context_length].get(extended_seq, 0)

    # Conditional probability calculation
    if context_freq > 0:
        return extended_freq / context_freq
    else:
        # Backoff to lower order model
        if context_length > 1:
            return calculate_probability(sequence[1:], char, tables)
        else:
            total_count = sum(tables[0].values())
            return tables[0].get(char, 0) / total_count if total_count > 0 else 0


def predict_next_char(sequence, tables, vocabulary):
    n = len(tables)

    # Trim sequence
    if len(sequence) >= n:
        sequence = sequence[-(n - 1):]

    # Initialize the probability dictionary
    probs = {}

    # Calculate probability for each character in the vocabulary
    for char in vocabulary:
        probs[char] = calculate_probability(sequence, char, tables)

    if not probs:
        return 'a'  # Default if no probabilities

    # Find the character with the maximum probability
    max_char = max(probs, key=probs.get)
    
    return max_char
