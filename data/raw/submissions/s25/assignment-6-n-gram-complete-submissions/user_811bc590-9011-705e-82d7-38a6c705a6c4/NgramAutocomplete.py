def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    table_list = []

    for i in range(n):
        table_list.append({})

    for current_char_index, char in enumerate(document):
        for n, table in enumerate(table_list):
            start_of_context_index = current_char_index-n
            if start_of_context_index >= 0:
                context = tuple(document[start_of_context_index:current_char_index])

                if char in table:
                    if context in table[char]:
                        table[char][context] += 1
                    else:
                        table[char][context] = 1
                else:
                   table[char] = {}
                   table[char][context] = 1

    return table_list


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
    sequence = sequence + char

    corpus_size = 0
    n = len(tables)

    for inner_table in tables[0].values():
        corpus_size += sum(inner_table.values())

    if corpus_size == 0 :
        return 0

    prev_freq = corpus_size

    total_probability = 1

    if n == 1:
        for char in sequence:
            curr_freq = tables[0].get(char, {}).get((), 0)
            total_probability *= (curr_freq / corpus_size)
        return total_probability


    for index, char in enumerate(sequence):
        if index >= n-1:
            start_of_context_index = index-n+1
            context = tuple(sequence[start_of_context_index:index])

            curr_freq = tables[n-1].get(char, {}).get(context, 0)

            probability = curr_freq / prev_freq
            total_probability *= probability

            #calculate the next freq
            start_of_context_index += 1
            context = tuple(sequence[start_of_context_index:index])
            prev_freq = tables[n-2].get(char, {}).get(context, 0)

            if prev_freq == 0:
                return 0
        else:
            context = tuple(sequence[0:index])

            curr_freq = tables[index].get(char, {}).get(context, 0)

            probability = curr_freq / prev_freq
            total_probability *= probability

            prev_freq = curr_freq

            if prev_freq == 0:
                return 0

    return total_probability




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

    max_probability = 0
    best_char = ''

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)

        if prob > max_probability:
            max_probability = prob
            best_char = char

    return best_char
