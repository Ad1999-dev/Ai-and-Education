import itertools


def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    def get_grams(chars, k):
        result = []
        lst = list(itertools.product(chars, repeat=k))
        for element in lst:
            if len(element) == k:
                result.append(''.join(element))
        return result

    def get_used_chars(document):
        return set(list(document))

    def substring_frequency(string, substring):
        return string.count(substring)

    used_chars = get_used_chars(document)
    tables = []
    for i in range(1, n+1):
        grams = get_grams(used_chars, i)
        table = {}
        for gram in grams:
            table[gram] = substring_frequency(document, gram)
        tables.append(table)

    # def get_event(str):
    #     return " \\mid ".join([f"{c}" for c in list(str)])

    # for table in tables:
    #     print(len(table))
    #     for key, value in table.items():
    #         print(f"| $P({get_event(key)})$ | $\\frac{{{value}}}{{{sum(table.values())}}}$ |")
    

    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.

    - **Returns**:
        - Returns a probability value for the sequence.
    """

    C = sum(tables[0].values())
    n = len(tables)
    # print(tables[-1])
    new_sequence = sequence + char
    result = tables[-1][new_sequence[:n]] / C
    for start in range(1, len(new_sequence)-n+1):
        numerator_str = new_sequence[start:start+n]
        denominator_str = numerator_str[:-1]
        numerator = tables[-1][numerator_str]
        denominator = tables[-2][denominator_str]
        result *= numerator / denominator
    return result


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

    result = 'a'
    maximum = 0
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        # print(sequence+char, prob)
        if prob > maximum:
            maximum = prob
            result = char

    return result
