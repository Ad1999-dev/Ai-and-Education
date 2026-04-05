import random
from collections import defaultdict

def create_frequency_tables(document, n):
    """
    Constructs a list of n frequency tables for an n-gram model.

    Parameters:
        document (str): The text document used to train the model.
        n (int): The number of grams for the n-gram model.

    Returns:
        List[Dict]: A list of n frequency tables.
    """
    tables = []

    for i in range(1, n + 1):
        table = defaultdict(lambda: defaultdict(int))
        for j in range(len(document) - i + 1):
            context = document[j:j + i - 1]
            char = document[j + i - 1]
            table[char][context] += 1
        tables.append(table)

    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given character after a sequence.

    Parameters:
        sequence (str): The sequence of characters as context.
        char (str): The character whose probability of occurrence after the sequence is to be calculated.
        tables (List[Dict]): The list of frequency tables.

    Returns:
        float: The probability value for the character following the sequence.
    """
    n = len(tables)
    k = min(len(sequence), n - 1)
    context = sequence[-k:] if k > 0 else ''
    table = tables[k]
    char_counts = table.get(char, {})
    count = char_counts.get(context, 0)
    total = sum(table[c].get(context, 0) for c in table)

    if total == 0:
        return 0.0
    return count / total


def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character based on the given sequence.

    Parameters:
        sequence (str): The sequence used as input to predict the next character.
        tables (List[Dict]): The list of frequency tables.
        vocabulary (Set[str]): The set of possible characters.

    Returns:
        str: The predicted next character.
    """
    max_prob = 0.0
    predicted_char = ''
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            predicted_char = char

    # If no character has a probability > 0, return a random character from the vocabulary
    if max_prob == 0.0:
        return random.choice(list(vocabulary))
    return predicted_char
