from collections import defaultdict

def create_frequency_tables(document, n):
    # Initialize tables for unigrams, bigrams, trigrams, etc.
    tables = []
    
    for i in range(1, n + 1):
        freq_table = defaultdict(int)
        for j in range(len(document) - i + 1):
            sequence = ''.join(document[j:j + i])  # Join characters to form a string for n-grams
            freq_table[sequence] += 1
        tables.append(freq_table)
    
    return tables


def calculate_probability(sequence, char, tables, vocabulary_size):
    # Extend the sequence by adding the next character
    extended_sequence = sequence + char
    n = len(extended_sequence)
    probability = 1.0

    # Probability of the first character (unigram level)
    first_char = extended_sequence[0]
    first_char_freq = tables[0].get(first_char, 0)
    if first_char_freq == 0:
        return 0  # Unseen character, probability is zero
    probability *= first_char_freq / sum(tables[0].values())

    # Iterate through each character in the sequence (starting from the second)
    for i in range(1, n):
        prefix = extended_sequence[:i]  # Prefix for conditional probability
        current_ngram = extended_sequence[:i + 1]  # Current n-gram

        # Get frequencies for the prefix and current n-gram
        prefix_freq = tables[i - 1].get(prefix, 0)
        current_ngram_freq = tables[i].get(current_ngram, 0)

        # If frequencies are missing, return 0
        if prefix_freq == 0 or current_ngram_freq == 0:
            return 0

        # Compute the conditional probability
        probability *= current_ngram_freq / prefix_freq

    return probability



def predict_next_char(sequence, tables, vocabulary):
    best_char = None
    best_prob = 0

    # Calculate the probability for each character in the vocabulary
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables, len(vocabulary))
        print(f"Sequence: {sequence}, Char: '{char}', Probability: {prob}")
        
        # Update the best character if the probability is higher
        if prob > best_prob:
            best_prob = prob
            best_char = char

    # Return the character with the highest probability
    return best_char if best_char is not None else next(iter(vocabulary))


