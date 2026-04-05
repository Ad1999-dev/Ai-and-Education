from collections import defaultdict
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

    for i in range(1, n+2):
        n_grams = defaultdict(int)

        for j in range(len(document) - i + 1):
            n_gram = document[j:j+i]
            n_grams[n_gram] += 1

        sorted_items = sorted(n_grams.items(), key=lambda item: item[1])
        sorted_grams = defaultdict(int, {key: value for key, value in sorted_items})
        
        tables.append(sorted_grams)
    
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
    full = sequence+char

    n = len(tables)-1
    joint_prob = 1.0
    size = sum(tables[0].values())

    p_x1 = tables[0][sequence[0]] / size
    joint_prob *= p_x1

    for i in range(1, len(full)+1):
        if i-n >= 0:
            numerator = full[i-n:i]
        else:
            numerator = full[:i]
        denominator = numerator[:-1]
        #print(numerator)
        #print(denominator)

        #get _gram tables of each part
        num_gram = len(numerator)
        den_gram = len(denominator)
        num_freq = tables[num_gram-1]
        den_freq = tables[den_gram-1]

        new_prob = (num_freq.get(numerator, 0) / den_freq[denominator] if den_freq[denominator] > 0 else 1)
        joint_prob *= new_prob

    return joint_prob


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
    max_prob = 0
    predicted_char = ""

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)

        if prob > max_prob:
            max_prob = prob
            predicted_char = char

    return predicted_char