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
    
    ngram_tables = []
    for i in range(1, n+2): #make n + 1 freq table as suggested by campus wire post 611
        ngram = defaultdict(int)  
        for j in range(len(document) - i + 1):  
            
            seq = document[j:j + i] 
            ngram[seq] += 1 
        ngram_tables.append(ngram)

    return ngram_tables


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
    ###Method 1 -> using ngram probability distribution
    n = len(tables) 
    full_sequence = sequence + char 
    prob = 1.0

    for i in range(len(full_sequence)):
        k = min(i, n - 1)
        context = full_sequence[i - k:i] 
        ngram = context + full_sequence[i]

        numerator = tables[k].get(ngram, 0)
        denominator = sum(tables[0].values()) if k == 0 else tables[k-1].get(context, 0)

        prob *= numerator / denominator

    return prob

    ### Method 2 -> general frequency case
    #predicting the next char given n+1 tables as suggested by campuswire post 611
    num = tables[len(sequence)].get(sequence + char, 0)
    dem = tables[len(sequence)-1].get(sequence)
    
    prob = num/dem
    
    return prob
    



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
    next_char = ''
    
    for char in vocabulary:
        curr_prob = calculate_probability(sequence, char, tables)
        if(curr_prob > max_prob):
            next_char = char
            max_prob = curr_prob
        
        
    
    return next_char

