from collections import defaultdict

def create_frequency_tables(document, n):

    # Initialize tables
    tables = [defaultdict(int)] + [defaultdict(lambda: defaultdict(int)) for _ in range(1, n)]
    
    # Loop over each position in document
    for i in range(len(document)):

        # Unigrams: each character is counted individually
        tables[0][document[i]] += 1

        # N-Grams: 2 to n
        for j in range(2, n + 1):

            # Check if enough characters are left to form an n-gram of length `j`
            if i + j <= len(document):
                # Separate context and the current character
                context = tuple(document[i:i + j - 1])
                current_char = document[i + j - 1]

                # Update frequency in table
                tables[j - 1][context][current_char] += 1

    return tables

def calculate_probability(sequence, char, tables):

    # Initialize n, probability, and context length
    n = len(tables)
    context_length = min(len(sequence), n - 1)
    probability = 1.0
    
    # Unigram Probability
    if context_length == 0:
        total_chars = sum(tables[0].values())
        count = tables[0].get(char, 0)

        if total_chars > 0:
            probability *= count / total_chars
        else:
            probability *= 0.0

    # Bigram and Higher Probability
    context = tuple(sequence[-context_length:])
    char_freq = tables[context_length].get(context, {}).get(char, 0)
    context_freq = sum(tables[context_length].get(context, {}).values())
    
    if context_freq > 0:
        probability *= char_freq / context_freq
    else:
        probability *= 0

    return probability


def predict_next_char(sequence, tables, vocabulary):

    # Initialize Max probability and predicted char
    max_prob = 0
    predicted_char = None

    # Calculate probability for each character in vocabulary, select max
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            predicted_char = char

    return predicted_char if predicted_char is not None else " "