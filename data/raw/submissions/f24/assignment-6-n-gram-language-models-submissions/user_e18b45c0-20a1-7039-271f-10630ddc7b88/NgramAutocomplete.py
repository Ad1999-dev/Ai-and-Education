from collections import defaultdict

def create_frequency_tables(document, n):
    """
    Constructs a list of frequency tables for an n-gram model, capturing character-based n-gram frequencies.
    """
    frequency_tables = [defaultdict(int) for _ in range(n)]
    for i in range(len(document)):
        for j in range(1, n + 1):
            if i + j <= len(document):
                sequence = document[i:i + j]
                frequency_tables[j - 1][sequence] += 1
    return frequency_tables

def calculate_probability(sequence, next_char, tables):
    """
    Calculates the probability of observing `next_char` following a given sequence of characters.
    This function now uses a fallback mechanism, starting from the highest possible n-gram.
    """
    # Determine length of the sequence and which n-gram to use
    for i in range(len(sequence), 0, -1):
        context = sequence[-i:]
        higher_ngram = context + next_char

        # Find the corresponding frequency table
        if i < len(tables):
            sequence_count = tables[i - 1].get(context, 0)
            higher_ngram_count = tables[i].get(higher_ngram, 0)

            # If we find a valid sequence_count, calculate the probability and return
            if sequence_count > 0:
                prob = (higher_ngram_count + 1) / (sequence_count + len(tables[i]))
                return prob

    # Fallback to unigram if no higher n-gram matches
    unigram_count = tables[0].get(next_char, 0)
    total_unigrams = sum(tables[0].values())
    return (unigram_count + 1) / (total_unigrams + len(tables[0]))

def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character based on the entire sequence using a fallback mechanism.
    """
    char_probs = []

    # Calculate the probability for each character in the vocabulary
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        char_probs.append((char, prob))

    # Sort character probabilities in descending order to find the top 5
    char_probs.sort(key=lambda x: x[1], reverse=True)

    # Select the character with the highest probability
    best_char = char_probs[0][0] if char_probs else " "
    return best_char

