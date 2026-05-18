from collections import defaultdict
import math

def create_frequency_tables(document, n):

    """
    This function constructs a list of `n` frequency tables for an n-gram model, each 
    table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = []

    for i in range(n+1):
        table_i = defaultdict(lambda: defaultdict(int))
        for char_index in range(len(document) - i): #subtract i in order to not exceed the size of the input text
            char_sequence = document[char_index : char_index + i + 1]
            context = char_sequence[:-1]
            char = char_sequence[-1]
            table_i[context][char] += 1
        tables.append(table_i)

    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the joint probability of the sequence followed by char using the n-gram frequency tables.

    Parameters:
    - sequence: A string of characters (length ≤ n-1).
    - char: The character we’re predicting.
    - tables: A list of frequency tables, from 0-gram up to n-gram.

    Returns:
    - Joint probability of observing sequence + char.
    """
    full_seq = sequence + char
    likelihood = 1.0
    n = len(tables) - 1  # because tables includes index 0 (unigram) to n

    for i in range(len(full_seq)):
        # Use up to n previous characters
        k = min(i, n)
        numerator_seq = full_seq[i - k:i + 1]  # context + current char
        denominator_seq = full_seq[i - k:i]    # just the context

        numerator_context = numerator_seq[:-1]
        numerator_char = numerator_seq[-1]

        table = tables[k]

        # Get frequency values 
        numerator_count = table[numerator_context][numerator_char]
        denominator_total = sum(table[denominator_seq].values())

        if denominator_total == 0:
            return 0  # Probability is zero if we’ve never seen this context

        likelihood *= numerator_count / denominator_total

    return likelihood



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
    best_char = None
    best_prob = 0 #default best set to negative infinity

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        print(prob)
        if prob > best_prob:
            best_prob = prob
            best_char = char

    if best_char == None or best_char == "":
        print("Returning None/empty string as best char!")
    return best_char
