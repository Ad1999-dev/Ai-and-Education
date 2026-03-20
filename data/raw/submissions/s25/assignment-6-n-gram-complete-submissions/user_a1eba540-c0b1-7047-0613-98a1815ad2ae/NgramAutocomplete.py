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
    # Create a list of empty frequency tables
    frequency_tables = [defaultdict(int) if i == 0 else defaultdict(lambda: defaultdict(int)) for i in range(n)]

    # Normalize document to lowercase to ensure case insensitivity
    document = document.lower()
    
    # Iterate through the document to create n-grams
    for i in range(len(document)):
        for j in range(1, n + 1):  # j is the current n-gram size
            if i + j <= len(document):
                ngram = document[i:i + j]
                if j == 1:
                    # Unigram: increment frequency for the character
                    frequency_tables[0][ngram] += 1
                else:
                    # Bigram, Trigram, etc.: increment frequency based on previous characters
                    prefix = ngram[:-1]  # All characters except the last one
                    frequency_tables[j - 1][ngram[-1]][prefix] += 1

    # Convert to probabilities
    for i in range(n):
       if i == 0:  # For unigrams, we need to normalize differently
            total_counts = sum(frequency_tables[0].values())
            for char in frequency_tables[0]:
                frequency_tables[0][char] /= total_counts if total_counts > 0 else 1
       else:
            for char in frequency_tables[i]:
                total_counts = sum(frequency_tables[i][char].values())
                for prefix in frequency_tables[i][char]:
                    frequency_tables[i][char][prefix] /= total_counts if total_counts > 0 else 1

    return frequency_tables

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
    n = len(tables)
    sequence_length = len(sequence)
    
    if sequence_length == 0:
        raise ValueError(f"The length of sequence must be between 0 and {n-1}.")
    
    # Use the exact sequence length for context
    table_index = min(sequence_length, n-1) # 0 for unigram, 1 for bigram, etc.
    prefix = sequence  # Use full sequence as context
    
    # Handle unigram case
    if table_index == 0:
        return tables[0].get(char, 0)
    
    # Handle higher-order n-grams
    freq = tables[table_index].get(char, {}).get(prefix, 0)
    total = sum(tables[table_index].get(c, {}).get(prefix, 0) for c in tables[table_index])
    
    return freq / total if total > 0 else 0


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
    context = sequence[-(len(tables)-1):] if len(tables) > 1 else ""
    probabilities = {}
    
    all_zero = True

    for char in vocabulary:
        prob = calculate_probability(context, char, tables)
        probabilities[char] = prob
        if prob > 0:
            all_zero = False

    # Fallback to unigram if all probabilities are 0
    if all_zero:
        for char in vocabulary:
            probabilities[char] = tables[0].get(char, 0)

    return max(probabilities, key=probabilities.get)
