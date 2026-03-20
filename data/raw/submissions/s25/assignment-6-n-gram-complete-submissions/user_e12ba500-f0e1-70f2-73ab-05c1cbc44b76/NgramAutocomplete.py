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
    frequency_tables_of_n_grams = []
    for i_th_table in range(1, n+1):
        i_th_gram_table = defaultdict(int)
        for position in range(len(document) - i_th_table + 1):
            sequence = document[position:position+i_th_table]
            i_th_gram_table[sequence] += 1
        frequency_tables_of_n_grams.append(i_th_gram_table)


    return frequency_tables_of_n_grams


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
    re_probability = 1
    corpus_size = sum(tables[0].values())
    if(len(sequence) == 0 or len(sequence) == 1):
        re_probability *= (tables[0].get(char,0)/corpus_size)
        return re_probability
    re_probability *= tables[0].get(sequence[0],0) / corpus_size
    sequence = sequence + char
    for pos in range(1,len(sequence)):
        current_letter = sequence[pos]
        previous_letters = sequence[max(0,pos - (len(tables)-1)):pos]
        table_correspoding_to_current_letter = tables[len(previous_letters)-1]
        freq_letters_so_far = tables[len(previous_letters)].get(previous_letters+current_letter,0)
        previous_letters_freq = table_correspoding_to_current_letter.get(previous_letters,0)
        if(previous_letters_freq == 0 or freq_letters_so_far == 0):
            return 0
        re_probability *= (freq_letters_so_far/previous_letters_freq)
        

    return re_probability


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
    init_max = 0
    init_char = ''
    for char in vocabulary:
        prob_char = calculate_probability(sequence,char,tables)
        if(prob_char> init_max):
            init_max = prob_char
            init_char = char
    print(init_max)
    return init_char
