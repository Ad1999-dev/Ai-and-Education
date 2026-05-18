def create_frequency_tables(document, n):
    """
    Constructs a list of `n` frequency tables for an n-gram model,
    each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = []
    unigram_counts = {}
    for char in document:
        unigram_counts[char] = unigram_counts.get(char, 0) + 1
    tables.append(unigram_counts)
    for i in range(2, n + 1):
        ngram_counts = {}
        for j in range(len(document) - i + 1):
            ngram = tuple(document[j:j + i - 1])
            next_char = document[j + i - 1]
            ngram_counts.setdefault(ngram, {}).setdefault(next_char, 0)
            ngram_counts[ngram][next_char] += 1
        tables.append(ngram_counts)

    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    n = len(tables)
    seq_len = len(sequence)
    if seq_len == 0:
        total_unigrams = sum(tables[0].values())
        return tables[0].get(char, 0) / total_unigrams if total_unigrams > 0 else 0
    relevant_table = tables[min(seq_len, n) - 1]
    prefix = tuple(sequence[max(0, seq_len - n + 1):])
    if prefix in relevant_table:
        total_following = sum(relevant_table[prefix].values())
        return relevant_table[prefix].get(char, 0) / total_following if total_following > 0 else 0
    else:
        return calculate_probability(sequence[1:], char, tables)

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
    probabilities = {}
    for char in vocabulary:
        probabilities[char] = calculate_probability(sequence, char, tables)

    if not probabilities:
        return None 

    return max(probabilities, key=probabilities.get)

