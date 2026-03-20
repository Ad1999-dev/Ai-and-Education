import random
from collections import defaultdict

def create_frequency_tables(document, n):
    """
    Constructs a list of n frequency tables for an n-gram model.

    Parameters:
    document (str): The text document used to train the model.
    n (int): The value of n for the n-gram model.

    Returns:
    list: A list of n frequency tables.
    """
    frequency_tables = [defaultdict(int) for _ in range(n)]

    for i in range(len(document)):
        for size in range(1, n + 1):
            if i + size <= len(document):
                ngram = document[i:i + size]
                frequency_tables[size - 1][ngram] += 1

    return frequency_tables


def calculate_probability(sequence, char, tables, alpha=0.01):
    """
    Calculates the probability of observing a given character after the sequence.

    Parameters:
    sequence (str): The sequence of characters.
    char (str): The character whose probability is to be calculated.
    tables (list): The list of frequency tables created by create_frequency_tables().
    alpha (float): Smoothing parameter to handle unseen n-grams.

    Returns:
    float: The probability of the character occurring after the sequence.
    """
    n = len(tables)
    sequence_length = len(sequence)

    # Truncate the sequence to (n-1) characters if it exceeds the model size
    if sequence_length >= n:
        sequence = sequence[-(n - 1):]

    table_index = min(sequence_length, n - 1)
    table = tables[table_index]

    # Calculate probabilities using the table
    ngram = sequence + char
    numerator = table.get(ngram, 0) + alpha
    denominator = sum(value + alpha for key, value in table.items() if key.startswith(sequence))

    if denominator == 0:
        return 0  # To avoid division by zero

    probability = numerator / denominator
    return probability


def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character based on the given sequence.

    Parameters:
    sequence (str): The sequence used as input to predict the next character.
    tables (list): The list of frequency tables.
    vocabulary (set): The set of possible characters.

    Returns:
    str: The predicted next character.
    """
    probabilities = {}

    # Calculate probability for each character in the vocabulary
    for char in vocabulary:
        probabilities[char] = calculate_probability(sequence, char, tables)

    # Normalize probabilities
    total_prob = sum(probabilities.values())
    if total_prob == 0:
        return random.choice(list(vocabulary))  # Fallback if all probabilities are zero

    normalized_probs = {char: prob / total_prob for char, prob in probabilities.items()}

    # Choose the character with the highest probability
    return max(normalized_probs, key=normalized_probs.get)