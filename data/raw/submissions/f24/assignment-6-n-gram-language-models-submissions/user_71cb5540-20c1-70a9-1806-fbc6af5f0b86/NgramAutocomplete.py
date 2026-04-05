def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    # Normalize the document
    document = document.replace('\n', ' ')
    characters = list(document)
    # Initialize frequency tables for each n-gram level
    frequency_tables = [{} for _ in range(n)]
    length = len(characters)

    # Count frequencies for each n-gram length from 1 to n
    for gram_length in range(1, n + 1):
        for i in range(length - gram_length+1):  # Changed this line
            n_gram = tuple(characters[i:i + gram_length])  # Removed + 1
            # The target character (the one we want to predict)
            char = characters[i + gram_length - 1]  # Changed this line
            # The preceding characters
            prev_chars = n_gram[:-1]

            # Initialize if character is not already in the frequency table
            if char not in frequency_tables[gram_length - 1]:
                frequency_tables[gram_length - 1][char] = {}

            # Initialize the context counter if it doesn't exist
            if prev_chars not in frequency_tables[gram_length - 1][char]:
                frequency_tables[gram_length - 1][char][prev_chars] = 0

            frequency_tables[gram_length - 1][char][prev_chars] += 1

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
    sequence = sequence+char
    seq_length = len(sequence)
    # Ensure that the sequence length is valid with respect to the tables
    if seq_length == 0 or seq_length >= len(tables):
        return 0.0  # Invalid input for calculating probability
    total_count_sequence = sum(
        tables[0][c].get((), 0) for c in tables[0] 
    )
    print(total_count_sequence)
    
    n_table = seq_length-1   # Corresponding table index (0-based)
    #print(tables)
    #if char not in tables[n_table]:
    #    return 0.0  # If character is not in the table, probability is 0
    probability = tables[0][sequence[0]].get(())
    prev_prob = probability
    probability = probability/total_count_sequence

    #print("PROB", probability)
    
    for i in range(1,seq_length):
        prev_chars = tuple(sequence[:i])  # Convert sequence to tuple for lookup
        curr_char = sequence[i]
        count_char_given_sequence = tables[i][curr_char].get(prev_chars)
        #print(count_char_given_sequence)
        if(count_char_given_sequence == None):
            return 0.0
        probability = probability*(count_char_given_sequence/prev_prob)
        prev_prob = count_char_given_sequence
    return probability

    # Retrieve the Context Frequency Table
    
    
    # Get the frequency of the character given the preceding context
    
    #print("count", count_char_given_sequence)
    # Total counts of the preceding sequence in the (n-1) frequency table
    #
    #print("total", total_count_sequence)
    # If total count is 0, we cannot calculate a probability
    #if total_count_sequence == 0:
    #    return 0.0
    
    # Calculate probability
    #probability = count_char_given_sequence / total_count_sequence
    
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
    max_prob = -100
    max_l = ''
    for l in vocabulary:
        prob = calculate_probability(sequence,l,tables)
        print(l,prob)
        if prob > max_prob:
            max_prob = prob
            max_l = l
    
    return max_l
