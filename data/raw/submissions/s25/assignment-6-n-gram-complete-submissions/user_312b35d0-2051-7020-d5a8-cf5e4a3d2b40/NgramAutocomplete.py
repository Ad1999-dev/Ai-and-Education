def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.
    - **Returns**:
        - Returns a list of n frequency tables.
    """
    all_tables = []
    # iterate over the ngram sizes
    for i in range(1,n+1):
        new_table = {}
        index = 0
        # index the start of the ngram
        while index+i <= len(document):
            ngram = document[index:index+i]
            if ngram not in new_table.keys():
                new_table[ngram] = 1
            else:
                new_table[ngram] += 1
            index += 1
        all_tables.append(new_table)
    return all_tables


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
    str = sequence+char
    total_len = sum(tables[0].values())
    n = len(tables)
    total_prob = 1
    unique_keys = {key for d in tables for key in d}

    for i in range(0, len(str)):
        if str[i:i+n] not in unique_keys:
            # print(f'{str[i:i+n]} not in unique_keys')
            return 0
        # the first term in the equation is always f(first character) / length of sequence
        if i == 0 or len(sequence)==1:
            total_prob *= tables[0][str[i]] / total_len
            # print(f"prob calculation 1: f({str[i]}) / {total_len} = {tables[0][str[i]]} / {total_len} = {tables[0][str[i]] / total_len}")
        # if we've processed <n letters then we need to use the <n tables
        elif i < n-1:
            total_prob *= tables[i][str[:i+1]] / tables[i-1][str[:i]]
            # print(f"prob calculation 2: f({str[:i+1]}) / f({str[:i]}) = {tables[i][str[:i+1]]} / {tables[i-1][str[:i]]} = {tables[i][str[:i+1]] / tables[i-1][str[:i]]}")
        # otherwise if we've gotten past n letters, then we take f(last n letters, inclusive of last letter) / f(last n letters, exclusive)
        else:
            total_prob *= tables[n-1][str[i-(n-1):i+1]] / tables[n-2][str[i-(n-1):i]]
            # print(f"prob calculation 3: f({str[i-(n-1):i+1]}) / f({str[i-(n-1):i]}) = {tables[n-1][str[i-(n-1):i+1]]} / {tables[n-2][str[i-(n-1):i]]}= {tables[n-1][str[i-(n-1):i+1]] / tables[n-2][str[i-(n-1):i]]}")
    return total_prob


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
    all_probs = {}
    i = 0
    for v in vocabulary:
        # calculate P(v | sequence) = P(sequence+v) / P(sequence)
        numerator = calculate_probability(sequence, v, tables)
        denominator =  calculate_probability(sequence[:-1], sequence[-1], tables)
        if denominator == 0:
            continue
        all_probs[v] = numerator/denominator
        # print(f'{v}: ', all_probs[v])
    # return key with max value
    if not all_probs:
        return ""
    return max(all_probs, key=all_probs.get)
