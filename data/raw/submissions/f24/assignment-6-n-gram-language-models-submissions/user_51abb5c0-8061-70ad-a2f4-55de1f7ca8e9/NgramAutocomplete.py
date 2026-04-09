from collections import defaultdict, Counter
def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    
    frequency_tables = [Counter() for _ in range(n)]
    
    # Iterate through the length of the document
    for i in range(len(document)): 
        # The inner for loop helps in extracting all the sequences of characters (unigrams, bigrams, etc.)
        for j in range(1, n+1):
            if i+j <= len(document):
                ngram = document[i:i+j]  # Use string directly to slice it 
                frequency_tables[j-1][ngram] += 1  # Update frequencies and count them based on that
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
    
    prob = 1.0  # Initializing probability to 1.0 so that we can multiply it later with conditional probabilities
    sequence_length = len(sequence)

    # Calculating the unigram probability for the first character in the sequence and taking that into account already
    first_char = sequence[0]  
    unigram_count = tables[0].get(first_char, 0)  
    total_unigrams = sum(tables[0].values())  

    if unigram_count == 0 or total_unigrams == 0:
        return 0.0  # If no unigrams found, probability is zero

    prob *= unigram_count / total_unigrams  # Initial probability for the unigram

    # Next we can calculate conditional probabilities for each subsequent character in the sequence and keep on updating it
    for i in range(1, sequence_length):
        subseq = sequence[:i]  # slicing it 
        next_char = sequence[i]  # to basically get the subsequence count for each part
        subseq_count = tables[i - 1].get(subseq, 0)  

        if subseq_count == 0:
            return 0.0  # If subsequence count is zero, return 0

        subseq_plus_char = subseq + next_char  
        subseq_plus_char_count = tables[i].get(subseq_plus_char, 0)   

        if subseq_plus_char_count == 0:
            return 0.0  
        
        prob *= subseq_plus_char_count / subseq_count  #updating the probability count

    # the last part is finally calculate the conditional probability for the next character  after the entire input sequence
    last_subseq = sequence
    last_subseq_count = tables[len(last_subseq) - 1].get(last_subseq, 0) #getting the counts

    if last_subseq_count == 0:
        return 0.0  # If no count for the entire sequence, return 0

    last_subseq_plus_char = last_subseq + char
    last_subseq_plus_char_count = tables[len(last_subseq)].get(last_subseq_plus_char, 0) if len(last_subseq) < len(tables) else 0

    if last_subseq_plus_char_count == 0:
        return 0.0  # If no count for the last sequence plus the next character, return 0

    prob *= last_subseq_plus_char_count / last_subseq_count

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
    if not vocabulary:
        return ''
    
    probabilities = []

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        probabilities.append((char, prob))
    
    probabilities.sort(key=lambda x: x[1], reverse=True)
    
    if probabilities:
        return probabilities[0][0] 
    else:
        return ''
    
    
    
    
    
    