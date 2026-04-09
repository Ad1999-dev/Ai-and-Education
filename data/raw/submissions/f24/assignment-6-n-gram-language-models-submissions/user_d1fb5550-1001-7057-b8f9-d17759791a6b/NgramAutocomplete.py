def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    freq_tables = [{} for _ in range(n+1)]

    for i in range(len(document)):
        for j in range(1, n + 2):
            if i + j <= len(document):
                n_gram = document[i: i + j]

                if n_gram in freq_tables[j - 1]:
                    freq_tables[j - 1][n_gram] += 1
                else:
                    freq_tables[j - 1][n_gram] = 1
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

    ''' if the length of the sequence is greater than the number of tables, take the last n chars from the sequence
     where n is the number of tables.'''
    # duplicate_seq = sequence
    # n = len(tables)

    # if(len(duplicate_seq) >= n):
    #     # take only last n chars
    #     x = (len(duplicate_seq) + 1) - n
    #     duplicate_seq = duplicate_seq[x:]

    # combined_seq = duplicate_seq + char

    # if(combined_seq in tables[len(combined_seq) - 1]):
    #     freq_combined_seq = tables[len(combined_seq) - 1][combined_seq]
    #     freq_seq = tables[len(duplicate_seq) - 1][duplicate_seq]
    #     return freq_combined_seq / freq_seq
    # else:
    #     return 0.0


    # calculating the initial probability i.e. f(x_1) / size(c)
    initial_char = sequence[0]
    freq_initial_char = tables[0][initial_char]
    size_corpus = sum(tables[0].values())

    probability = freq_initial_char / size_corpus

    # multiplying the frequencies of next subsequences (numerator)
    for i in range(1, len(sequence) + 1):
        curr_subseq = sequence[:i]
        next_subseq = curr_subseq + char if i == len(sequence) else sequence[:i + 1]
        if next_subseq in tables[i]: 
            freq_next_subseq = tables[i][next_subseq]
        else: 
            freq_next_subseq = 0

        probability *= freq_next_subseq

    # dividing the probability by the denominator term
    for i in range(1, len(sequence) + 1):
        curr_seq = sequence[:i]
        freq_curr_seq = tables[i - 1][curr_seq]

        if freq_curr_seq == 0:
            return 0 # avoid the case of dividing by zero.
        probability /= freq_curr_seq
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
    maximum_prob = -1
    char_with_max_prob = ''
    for char in vocabulary:
        probability = calculate_probability(sequence, char, tables)
        if probability > maximum_prob:
            maximum_prob = probability
            char_with_max_prob = char
    return char_with_max_prob

