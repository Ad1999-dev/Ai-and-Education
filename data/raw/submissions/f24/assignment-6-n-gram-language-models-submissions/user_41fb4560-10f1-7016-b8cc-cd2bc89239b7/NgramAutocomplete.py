def create_frequency_tables(document: str, n: int):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    
    tables: list[dict[str, int]] = [{} for i in range(n)]
    doclen = len(document)
    
    for i in range(doclen):
        for j in range(n):
            if i + j < doclen:
                ngram: str = document[i:i + j + 1]
                
                if ngram in tables[j]:
                    tables[j][ngram] += 1
                else:
                    tables[j][ngram] = 1
    
    return tables


def calculate_probability(sequence: str, char: str, tables: list[dict[str, int]]):
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
    seq_len = len(sequence)

    probability = 1.0

    for i in range(seq_len, 0, -1):
        if i < n:  
            ngram_freq = tables[i].get(sequence + char, 0)
            total_count = sum(tables[i].values())
            
            if total_count == 0:
                return 0
            
            probability *= ngram_freq / total_count
            break

        ngram_freq = tables[n - 1].get(sequence[i - (n - 1):] + char, 0)
        prefix_freq = tables[n - 2].get(sequence[i - (n - 1):], 0)
        
        if prefix_freq == 0:
            return 0
        
        probability *= (ngram_freq / prefix_freq)
        char = sequence[-1]
        sequence = sequence[:-1]
    
    return probability


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

    return max((calculate_probability(sequence, c, tables), c) for c in vocabulary)[1]
