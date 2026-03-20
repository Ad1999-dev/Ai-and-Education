"""
This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

- **Parameters**:
    - `document`: The text document used to train the model.
    - `n`: The number of value of `n` for the n-gram model.

- **Returns**:
    - Returns a list of n frequency tables.
"""
def create_frequency_tables(document, n):
    tables = []
    for _ in range(n):
        tables.append({})

    for i, char in enumerate(document):
        for k in range(1, n + 1): # k = 1 to n
            if i - (k - 1) < 0:
                break

            prev = document[i - (k - 1) : i] # the previous k-1 characters
            table = tables[k - 1]

            if char not in table: # if this character is not in the table, create a new entry
                table[char] = {}

            table[char][prev] = table[char].get(prev, 0) + 1

    return tables

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
    for k in range(min(len(sequence), len(tables)-1), -1, -1): # Iterate over the tables in reverse order to find the longest history
        if k > 0:
            hist = sequence[-k:] 
        else: 
            hist = ""

        table = tables[k]                              
        numerator = table.get(char, {}).get(hist, 0) # counts of hist for the given char
        denominator = 0 # total counts for hist across all possible next chars

        for c_dict in table.values():
            denominator += c_dict.get(hist, 0)

        if denominator > 0:
            return numerator / denominator
    return 0 # no match found

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
    best_char = None
    best_prob = -1.0

    for c in vocabulary:
        prob = calculate_probability(sequence, c, tables)
        if prob > best_prob:
            best_prob = prob
            best_char = c

    return best_char
