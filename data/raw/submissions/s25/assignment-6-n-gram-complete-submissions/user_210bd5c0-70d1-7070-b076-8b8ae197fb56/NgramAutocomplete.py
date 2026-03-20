def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    
    frequency_tables = []
    
    #for i in range(1, n)
    for i in range(1, n+1):
    #split into i-character long chunks
        chunks = [document[j:j+i] for j in range(0, len(document), i)]
    #get setified version
        n_grams = set(chunks)
    #for each i-gram in the set, find it's frequency
        freq_table = {}
        for gram in n_grams:
            freq_table[gram] = document.count(gram)
        frequency_tables.append(freq_table)

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
   
    #get length of tables aka the n in n-gram
    #get the last n-1 letters
    #get the total frequency of the grams with sequence as their prefix
    #divide by the 
    n = len(tables)
    table = tables[n-1]
    total = 0
    if n == 1:
        new_seq = char
        for (gram, count) in table.items():
            total += count
        return table.get(new_seq)/total
    
    new_seq = sequence[-n+1:]
   
    for (gram, count) in table.items():
        if gram.startswith(new_seq):
            total += count
    if table.get(new_seq+ char) != None:
        return table.get(new_seq + char)/total
    else:
        return 0




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
    
    max_char = ""
    max_char_prob = 0
    #for each char in vocab
    for char in vocabulary:
        char_prob = calculate_probability(sequence, char, tables)
    #if char's probability is greater than the character with the current max probability
        if char_prob > max_char_prob:
            max_char = char
            max_char_prob = char_prob
    #then this char has the new max probability 
    return max_char

