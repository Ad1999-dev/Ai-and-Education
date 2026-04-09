from collections import defaultdict

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
     # Initialize a list of n empty dictionaries
    freq_tables = [{} for _ in range(n)]

    for i in range(len(document)):
        # Table for 1-grams (individual characters)
        char = document[i]

        # Skip single-character space
        if char == ' ':
            continue

        if char in freq_tables[0]:
            freq_tables[0][char] += 1
        else:
            freq_tables[0][char] = 1

        # Tables for 2-grams to n-grams
        for j in range(2, n + 1):
            if i - j + 1 >= 0:
                ngram = document[i - j + 1 : i + 1]

                 # Skip n-grams that contain a space
                if ' ' in ngram:
                    continue

                if ngram in freq_tables[j - 1]:
                    freq_tables[j - 1][ngram] += 1
                else:
                    freq_tables[j - 1][ngram] = 1

    return freq_tables



def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    n = len(tables)  # e.g., if 3 tables: 1-gram, 2-gram, 3-gram

    # Truncate sequence to length n - 1 for matching the right table
    context = sequence[-(n-1):] if n > 1 else ''

    # Get the appropriate frequency table
    table_index = len(context)  # Table for (context + char) is at table_index
    if table_index >= len(tables):
        return 0.0  # If context is too short, cannot compute

    # Form the extended sequence: context + char
    extended = context + char

    # Get counts from the appropriate tables
    full_count = tables[table_index].get(extended, 0)  # numerator
    context_count = tables[table_index - 1].get(context, 0)  # denominator

    # Avoid division by zero
    if context_count == 0:
        return 0.0

    return full_count / context_count


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
    max_prob = -1.0
    best_char = None

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char

    # Fallback if no character found (e.g., context unseen)
    return best_char if best_char is not None else ' '  # or random.choice(list(vocabulary))
