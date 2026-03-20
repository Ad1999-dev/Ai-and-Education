from collections import defaultdict
def create_frequency_tables(document, n):
   
        

    tables = []
    
    for i in range(n):
        dictt = defaultdict(int)
        for j in range(len(document) - i):
            ngram = document[j:j+i+1]
            dictt[ngram] += 1
        tables.append(dict(dictt))  # convert to regular dict here if desired
    
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

    seq_len_index = len(sequence)-1

    if seq_len_index >= len(tables):
        sequence = sequence[-(len(tables) - 1):]  
        seq_len_index = len(sequence) - 1  
    
    
    denominator_count = tables[seq_len_index].get(sequence, 0)
    

    to_find_word = sequence + char
    ngram_index = len(to_find_word) - 1

    if ngram_index >= len(tables):
        return 
    
    numerator_count = tables[ngram_index].get(to_find_word, 0)

    if denominator_count == 0:
        probability = 0
    else:
        probability = numerator_count/denominator_count

    return probability


def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character based on the given sequence.

    - **Parameters**:
        - `sequence`: The sequence used as input to predict the next character.
        - `tables`: The list of frequency tables.
        - `vocabulary`: The set of possible characters.
    
    - **Functionality**:
        - Calculates the probability of each possible next character in the vocabulary, using `calculate_probability()`.
Microsoft.QuickAction.BatterySaver
    - **Returns**:
        - Returns the character with the maximum probability as the predicted next character.
    """
    max_probability = 0
    predicted_char = None

    seq_length = len(sequence)

    if seq_length > len(tables):
        sequence = sequence[-(len(tables)-1):]

    

    for char in vocabulary:
        our_seq = sequence + char
        probability = calculate_probability(sequence, char , tables)
        probability = probability if probability is not None else 0 
        if probability > max_probability:
            max_probability = probability
            predicted_char = char

    if predicted_char is None:
        predicted_char = " "
    return predicted_char
