def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    # creates {prefix1: {char1: count, char2: count, etc.}, etc.}
    tables = []
    for n_i in range(1, n+1):
        table = {}
        for j in range(len(document) - n_i + 1):
            prefix = document[j : j + n_i - 1]
            cur = document[j + n_i - 1]
            if prefix == "": # no prefix so just the frequency of the letters
                if cur not in table:
                    table[cur] = 0
                table[cur] += 1
            else: # has a prefix so index by the prefix
                if prefix not in table:
                    table[prefix] = {}
                if cur not in table[prefix]:
                    table[prefix][cur] = 0
                table[prefix][cur] += 1
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

    # calc P(sequence + char)
    prob = 1
    max_n = len(tables) # since we only have tables that go up to n
    chars = sequence + char
    for t in range(len(chars)):
        n = min(t + 1, max_n) # either use the max n or the number of characters we have so far
        cur = chars[t]
        prefix = chars[t - n + 1 : t] # get the n - 1 characters before the current char
        if n == 1: # if there is no prefix before our character (first term)
            size_C = sum(tables[0].values())
            if cur in tables[0]:
                prob *= (tables[0][cur] / size_C)
            else:
                return 0 # if the letter is not in the table, we have not seen it in the corpus so our prob is 0
        else:
            if prefix in tables[n - 1] and cur in tables[n - 1][prefix]:
                # calc f(chars) / f(prefix)
                f_char = tables[n - 1][prefix][cur] # f(chars)
                f_prefix = sum(tables[n - 1][prefix].values()) # f(prefix)

                prob *= (f_char / f_prefix)
            else:
                return 0 # if we have not seen those combination of chars in our corpus
    return prob

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
    highest_prob = 0
    best_char = ""
    for char in vocabulary:
        cur_prob = calculate_probability(sequence, char, tables)
        if cur_prob > highest_prob:
            highest_prob = cur_prob
            best_char = char
    return best_char
