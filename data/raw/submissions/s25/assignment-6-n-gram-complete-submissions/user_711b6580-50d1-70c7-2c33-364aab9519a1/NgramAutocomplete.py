def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = []
    
    for i in range(1, n+1):
        table = {}

        for j in range(len(document) - i + 1):
            ngram = document[j:j+i]

            if i == 1: 
                if ngram in table:
                    table[ngram] += 1
                else: 
                    table[ngram] = 1
            else: 
                prefix = ngram[:-1]
                suffix = ngram[-1]

                if prefix not in table: 
                    table[prefix] = {}
                if suffix in table[prefix]:
                    table[prefix][suffix] += 1
                else:
                    table[prefix][suffix] = 1
        tables.append(table)
    return tables


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
    if not sequence:
        # If char is in unigram table
        if char in tables[0]:
            # Total character occurrences in the corpus
            total_chars = sum(tables[0].values())
            # Occurrences of the specific character
            char_count = tables[0][char]
            
            # Apply smoothing
            vocab_size = len(tables[0])
            return (char_count + 1) / (total_chars + vocab_size)
        else:
            # Character not in vocabulary, apply smoothing
            total_chars = sum(tables[0].values())
            vocab_size = len(tables[0])
            return 1 / (total_chars + vocab_size)
    
    # For longer sequences
    context_length = len(sequence)
    
    # If sequence is longer than our n-gram model supports
    if context_length >= len(tables):
        # Use only the last n-1 characters as context
        sequence = sequence[-(len(tables)-1):]
        context_length = len(sequence)
    
    # Check if this context exists in the appropriate table
    if sequence in tables[context_length]:
        # Check if the character appears in this context
        if char in tables[context_length][sequence]:
            # Get the count of this character in this context
            char_count = tables[context_length][sequence][char]
            # Get the total occurrences of this context
            total_count = sum(tables[context_length][sequence].values())
            
            # Apply smoothing
            vocab_size = len(tables[0])
            return (char_count + 1) / (total_count + vocab_size)
        else:
            # Character not seen in this context, apply smoothing
            total_count = sum(tables[context_length][sequence].values())
            vocab_size = len(tables[0])
            return 1 / (total_count + vocab_size)
    else:
        # Context not found, back off to lower-order model
        if context_length > 1:
            return calculate_probability(sequence[1:], char, tables)
        else:
            # Already at lowest order, use unigram probability
            return calculate_probability("", char, tables)

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
    max_prob = -1
    next_char = None
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)

        if prob > max_prob:
            max_prob = prob
            next_char = char
    return next_char
