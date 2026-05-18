from utilities import read_file, print_table


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
        for j in range(1, n+1):
            if i + j <= length:
                ngram = document[i:i + j]
                if ngram in frequency_tables[j-1]:
                    frequency_tables[j-1][ngram] += 1
                else:
                    frequency_tables[j-1][ngram] = 1
    # print(len(frequency_tables), "This is the length of the freq table")
    # print(frequency_tables, "this is the freq table")
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
    s = sequence + char
    ratio = 1.0

    first_char_count = tables[0].get(s[0], 0)
    total_unigrams = sum(tables[0].values())
    if total_unigrams == 0 or first_char_count == 0:
        return 0.0

    ratio *= first_char_count / total_unigrams
    
    for i in range(1, len(s)):
        current_ngram = s[:i + 1]
        previous_ngram = s[:i]
    
        current_count = tables[i].get(current_ngram, 0)
        previous_count = tables[i - 1].get(previous_ngram, 0)   

        if previous_count == 0:
            print(f"Previous count is 0 for '{previous_ngram}'")
            return 0.0

        ratio *= current_count / previous_count

        if ratio == 0:
            print(f"Ratio became 0 at step {i}")
            print(f"This is table {i}, {tables[i]}")
    
    return ratio


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

    highestPercent = 0
    curChar = ""

    
    for char in vocabulary:
        t = calculate_probability(sequence, char, tables)
        # print(t, "this is t")

        if (highestPercent < t):
            print("reached here")
            highestPercent = t
            curChar = char

    # print(curChar)
    # print(highestPercent, "this is the highest percent")
    return curChar

