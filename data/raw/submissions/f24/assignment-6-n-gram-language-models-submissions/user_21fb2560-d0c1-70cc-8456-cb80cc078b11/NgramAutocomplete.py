"""
This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

- **Parameters**:
    - `document`: The text document used to train the model.
    - `n`: The number of value of `n` for the n-gram model.

- **Returns**:
    - Returns a list of n frequency tables.
"""


def create_frequency_tables(document, n):
    returnList = []
    document_length = len(document)

    for i in range(1, n + 1):
        table = {}

        # Iterate through the document, considering sequences of length `i`
        for j in range(document_length - i + 1):
            # Get i-length sequence from the document
            sequence = document[j:j + i]

            if sequence in table:
                table[sequence] += 1
            else:
                table[sequence] = 1

        # Append the frequency table to the return list
        returnList.append(table)

    return returnList

# Initially, I thought that this function returned tables consisting of all possible combinations (sizes 1 to n)
# of the vocabulary regardless of whether thee sequences actually existed in the text or not
# Then I realized that it's only supposed to count up for the sequences that actually exist in text
# and thus implemented the function above
#     vocabulary = []
#     returnList = []
#     for char in document:
#         if char in vocabulary:
#             continue

#         vocabulary.append(char)

#     for i in range(1, n+1):  # excluding n+1
#         table = {}
#         sequence = ''
#         for index in range(0, len(vocabulary)):
#             # have to use 'i' in some way here (while len(sequence) <= i)?
#             sequence = sequence + vocabulary[index]
#             for sequence in document:
#                 if sequence in table:
#                     table[sequence] += 1
#                 if sequence not in table:
#                     table[sequence] = 1

#         returnList.append[table]

#     return returnList


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
    sequence_length = len(sequence)
    n = len(tables)
    probability = 0
    word = sequence + char

    # since document isn't passed on here, we need to calculate the length of the document here
    doc_length = 0
    for key in tables[0]:
        doc_length += tables[0][key]

    if sequence_length == 0:
        first_table = tables[0]
        freq = first_table[char]
        probability = freq/doc_length
        return probability

    if sequence_length >= n:
        # probability of char given the last n-1 things before char = f(seq + char)/f(seq) = f(n chars)/f(n-1 chars)
        word_table = tables[n - 1]
        seq_table = tables[n - 2]
        # word = last n letters in sequence + char
        word = word[-n:-1] + word[-1]
        # frequency of last n letters in sequence + char
        if word not in word_table:
            f_word = 0
        else:
            f_word = word_table[word]
        # last n letters in sequence + char, excluding the char itself
        if word[-n:-1] not in seq_table:
            f_seq = 0
        else:
            f_seq = seq_table[word[-n:-1]]

        if f_seq == 0:
            probability = 0
        else:
            probability = f_word/f_seq

    else:
        # since first index of tables is 0, which contains a table of length 1 seqs
        word_table = tables[sequence_length]
        seq_table = tables[sequence_length - 1]

        if word not in word_table:
            f_word = 0
        else:
            f_word = word_table[word]

        if sequence not in seq_table:
            f_seq = 0
        else:
            f_seq = seq_table[sequence]

        if f_seq == 0:
            probability = 0
        else:
            probability = f_word/f_seq

    return probability


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
    best_prob = 0
    return_char = ''
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)

        if prob > best_prob:
            best_prob = prob
            return_char = char

    return return_char
