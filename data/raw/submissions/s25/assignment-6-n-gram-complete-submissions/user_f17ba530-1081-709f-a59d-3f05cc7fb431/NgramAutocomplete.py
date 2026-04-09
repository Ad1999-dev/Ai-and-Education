def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables = [{} for _ in range(n+1)]
    length = len(document)

    for i in range(length):
        for subNIndex in range(1, n + 2):
            if i + subNIndex <= length:
                n_gram = document[i + subNIndex - 1]
                if subNIndex == 1:
                    preceding_context = ""
                else:
                    preceding_context = document[i:i + subNIndex - 1]
                # key = (preceding_context, n_gram)
                # if key in frequency_tables[subNIndex - 1]:
                #     frequency_tables[subNIndex - 1][key] += 1
                # else:
                #     frequency_tables[subNIndex - 1][key] = 1
                key = n_gram
                if key not in frequency_tables[subNIndex - 1]:
                    frequency_tables[subNIndex - 1][key] = {}

                if preceding_context in frequency_tables[subNIndex - 1][key]:
                    frequency_tables[subNIndex - 1][key][preceding_context] += 1
                else:
                    frequency_tables[subNIndex - 1][key][preceding_context] = 1

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
    seqLength = len(sequence)
    if seqLength > n or seqLength < 0:
        return 0.0
    # frequency_table = tables[seqLength - 1]
    frequency_table = tables[seqLength]

    if seqLength > 0:
        preceding_context = sequence[-(seqLength):]
        # key = (preceding_context, char)
        key = char
    else:
        preceding_context = ""
        key = char

    count_key = 0
    
    if key not in frequency_table:
        count_key = 0
    else:
        count_key = frequency_table[key].get(preceding_context, 0)

    total_count = 0
    if(seqLength == 0):
        for k in frequency_table.keys():
            for num in frequency_table[k].keys():
                total_count += frequency_table[k][num]
    else:
        prev_table = tables[seqLength - 1]
        prev_key = preceding_context[-1:]
        prev_context = preceding_context[:-1]
        if prev_key in prev_table:
            total_count = prev_table[prev_key].get(prev_context, 0)

    if total_count == 0:
        return 0.0
    
    probability = count_key / total_count
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
    max_probability = 0.0
    predicted_char = ""
    
    # Loop through each character in the vocabulary to calculate its probability
    for char in vocabulary:
        # print(sequence)
        # print(char)
        # Calculate the probability of char given the sequence
        probability = calculate_probability(sequence, char, tables)
        # Check if this probability is the highest found so far
        if probability > max_probability:
            max_probability = probability
            predicted_char = char
    
    return predicted_char
