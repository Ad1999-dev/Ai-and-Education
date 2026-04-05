def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
    document_length = len(document)
    for i in range(document_length):
        for j in range(min(n, document_length)):
            if i >= j:  #ensure we have enough characters for the current n-gram
                prev_chars = document[i - j:i]  #extract the previous j characters
                current_char = document[i]      #current character
                tables[j][current_char][prev_chars] += 1
    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    sequence_length = len(sequence)
    
    if sequence_length == 0:  #unigram case
        total_count = sum(sum(counts.values()) for counts in tables[0].values())
        char_count = sum(tables[0][char].values()) if char in tables[0] else 0
        return char_count / total_count if total_count > 0 else 0

    #use the table corresponding to the sequence length
    table_index = min(sequence_length, len(tables) - 1)
    table = tables[table_index]

    prev_context = sequence[-table_index:]  #getting previous context based on the table index
    total_count = sum(
        table[c][prev_context] for c in table.keys() if prev_context in table[c]
    )
    char_count = table[char][prev_context] if prev_context in table[char] else 0

    if total_count == 0:  #to avoid division by zero
        return 0  #probability is zero if there is no total count

    return char_count / total_count


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
    sequence_length = len(sequence)
    n = len(tables)
    context = sequence[-(n - 1):]

    probabilities = {}
    for char in vocabulary:
        probabilities[char] = calculate_probability(context, char, tables)
    
    #find the character with the highest probability
    most_likely_char = max(probabilities, key=probabilities.get)
    return most_likely_char
