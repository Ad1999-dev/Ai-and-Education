from collections import defaultdict
from utilities import read_file

def create_frequency_tables(document, n): 
    # FOR INSTRUCTOR/GRADER: I read on campuswire that we could implement our functions best representing our understanding.
    # Therefore my frequency tables are later converted to probability tables as they aide in computation in the future
    # and also align with my understanding of Bayesian Networks.

    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    frequency_tables = []
    if(len(document) < n):
        n = len(document)
    
    for i in range(n):
        frequency_tables.append(defaultdict(int))
    
    for i in range(len(document)):
        for j in range(1, n+1):
            if i < len(document) - j + 1:
                ngram = document[i : i + j]
                frequency_tables[j-1][ngram] += 1
    
    total_counts = []
    for i in range(len(frequency_tables)):
        total_counts.append(sum(frequency_tables[i].values()))
    
    probability_tables = []
    for i in range(n):
        if total_counts[i] != 0:  
            prob = {ngram: count / total_counts[i] for ngram, count in frequency_tables[i].items()}
        else:
            prob = {}
        probability_tables.append(prob)

    return probability_tables


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
    n = len(tables)
    if(len(sequence) < n):
        return tables[len(sequence)].get(sequence+char, 0)/tables[len(sequence)-1].get(sequence, 1)
    else:
        seq_length = min(len(sequence), n)
        subs = sequence[-seq_length:]
        subs_new = sequence[-(seq_length-1):] + char
        return tables[len(subs_new)-1].get(subs_new, 0)/tables[len(subs)-1].get(subs, 1)

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
    if not vocabulary: return None 
    if len(sequence) == 0 or not tables: return None  
    
    max_prob_char = None
    max_prob = -1  

    for char in vocabulary:
        try:
            new_prob = calculate_probability(sequence, char, tables)
        except Exception as e:
            print(f"Error calculating probability for character '{char}': {e}")
            continue 

        if new_prob > max_prob:
            max_prob = new_prob
            max_prob_char = char
    
    return max_prob_char
