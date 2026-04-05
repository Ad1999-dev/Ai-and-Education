import itertools

"""
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
def create_frequency_tables(document, n):
    output = []

    alphabet = [] # a list of all of the unique characters in the document
    for c in document:
        if alphabet.count(c) == 0:
            alphabet.append(c)
    alphabet.sort()

    # Make the whole table, and all of the frequencies are 0
    for i in range(n+1):
        frequency_table = {}
        for letter in alphabet:
            prev_chars_dict = {}
            for combination in itertools.product(alphabet, repeat=i):
                prev_chars_dict[''.join(combination)] = 0
            frequency_table[letter] = prev_chars_dict
        output.append(frequency_table)

    # Add all the frequencies by going through the document
    for i in range(len(document)):
        for j in range(n + 1):
            if i + j < len(document):
                output[j][document[i+j]][document[i: i + j]] += 1

    return output

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
    # When the sequence length is larger than the n-grams, we use the last n chars of the sequence
    if(len(sequence) > len(tables)): 
        numerator = tables[len(tables)][char][sequence[len(sequence)-len(tables):len[sequence]-1]]
        denominator = tables[len(tables)][sequence[-1]][sequence[len(sequence)-len(tables)-1:len(sequence)-1]]
    else:
        numerator = tables[len(sequence)][char][sequence]
        denominator = tables[len(sequence)-1][sequence[-1]][sequence[0:len(sequence)-1]]
    if (denominator) == 0:
        return 0
    return numerator / denominator

    # # Calculate Size(C)
    # C = 0
    # for keys in tables[0]:
    #     C += tables[0][keys]['']

    # # if the sequence is empty
    # if len(sequence) == 0:
    #     return tables[0][char][''] / (C) # f(x_1)/size(C)
    
    # if tables[0][sequence[len(sequence)-1]][''] == 0:
    #     return 0
    # return (tables[0]) / (tables[0][sequence[len(sequence)-1]][''])

    # # calculating the joint probability
    # prob_denom = tables[0][sequence[0]][''] / (C) # f(x_1)/size(C)
    # prob_denom *= (tables[1][char][sequence[0]]) / (tables[0][sequence[0]][''])
    # # When sequence is more than 1 char
    # for i in range(len(sequence) - 1):
    #     prob_denom *= ((tables[1][sequence[i+1]][sequence[i]])/(tables[0][sequence[i]]['']))
    #     print(prob_denom)

    # calculating the frequency of the denominator
    # prob_len = len(tables) if len(sequence) > len(tables) else len(sequence)
    # freq_denom = 0
    # for c in tables[prob_len]:
    #     freq_denom += tables[prob_len][c][sequence]

    # if freq_denom == 0:
    #     return 0
    # print(str(freq_denom) + ', ')
    # return ((tables[prob_len][char][sequence])/freq_denom)


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
    current_max_probability = 0
    current_next_char = ''
    for char in vocabulary:
        if calculate_probability(sequence, char, tables) > current_max_probability:
            current_next_char = char
            current_max_probability = calculate_probability(sequence, char, tables)
    return current_next_char
