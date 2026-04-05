"""
This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

- **Parameters**:
    - `document`: The text document used to train the model.
    - `n`: The number of value of `n` for the n-gram model.

- **Returns**:
    - Returns a list of n frequency tables.
"""
def create_frequency_tables(document, n):
    listOfFrequencyTables = []
    for i in range(1, n + 1):
        freqTable = {}
        for j in range(len(document) - i + 1): # sets the bounds of the n gram, which is length - n + 1
            n_gram = document[j:j + i]  # gets the n-gram at each level of n
            if n_gram in freqTable:
                freqTable[n_gram] += 1  
            else:
                freqTable[n_gram] = 1
                
        listOfFrequencyTables.append(freqTable)  
    return listOfFrequencyTables


"""
Calculates the probability of observing a given sequence of characters using the frequency tables.

- **Parameters**:
    - `sequence`: The sequence of characters whose probability we want to compute.
    - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
    - `char`: The character whose probability of occurrence after the sequence is to be calculated.

- **Returns**:
    - Returns a probability value for the sequence.
"""
def calculate_probability(sequence, char, tables):
    # P(char | sequence) = P(sequence N char) / P(sequence)
    # say sequence is 'abc' -> P(sequence) = P(a) * P(b | a) * P(c | ab)
        # generalize this further

    # Calculate P(sequence)
    seqProb = 1.0
    n = len(tables)

    for i in range(len(sequence)):
        current_context = sequence[max(0, i - (n - 1)): i]  # Get the relevant context
        n_gram = current_context + sequence[i]

        # Ensure n_gram only accesses valid tables
        if len(n_gram) > n:
            continue  # Skip or handle if context is larger than tables available

        joint_freq = tables[len(n_gram) - 1].get(n_gram, 0)
        context_freq = tables[len(current_context)].get(current_context, 0)

        # Adjust probability calculations
        if context_freq > 0:
            prob = joint_freq / context_freq
        else:
            prob = 0  # Respect zero context frequency

        if prob > 0:
            seqProb *= prob  # Only multiply when probability is non-zero
        else:
            seqProb = 0  # If zero, no point continuing

    # Now calculate P(char | sequence)
    full_n_gram = (sequence[-(n - 1):] + char) if len(sequence) >= (n - 1) else (sequence + char)
    joint_freq_with_char = tables[len(full_n_gram) - 1].get(full_n_gram, 0)
    context_freq_for_sequence = tables[len(sequence) - 1].get(sequence, 0)

    # Handle zero probabilities cleanly
    if context_freq_for_sequence > 0:
        char_probability = joint_freq_with_char / context_freq_for_sequence
    else:
        char_probability = 0

    return char_probability


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
def predict_next_char(sequence, tables, vocabulary):
    max_prob = -1 
    best_char = None  

    for char in vocabulary:
        char_probability = calculate_probability(sequence, char, tables)
        
        if char_probability > max_prob:
            max_prob = char_probability
            best_char = char

    return best_char
