from sklearn.feature_extraction.text import CountVectorizer
from utilities import read_file
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
    corpus = [document]
    for i in range(1, n + 1):
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(i, i))
        X = vectorizer.fit_transform(corpus)
        frequency_table = X.toarray().sum(axis=0)
        ngram_labels = vectorizer.get_feature_names_out()
        
        frequency_dict = {ngram: count for ngram, count in zip(ngram_labels, frequency_table)}
        tables.append(frequency_dict)

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
    seq_plus_char = sequence + char
    n = len(seq_plus_char)

    if n == 0 or n > len(tables):
        return None
    
    # Technically I only need to do one calculation P('abcd') = f(abcd)/size(C) 
    # But I'll use the chain rule for the sheer thrill
    
    prev_freq = sum(tables[0].values())
    probability = 1
    for i in range(n):
        if prev_freq == 0:
            return 0
        else:
            gram = seq_plus_char[:i+1]
            freq = int(tables[i].get(gram)) if tables[i].get(gram) is not None else 0
            probability = probability * float(freq/prev_freq)
            prev_freq = freq
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

    - **Returns**:
        - Returns the character with the maximum probability as the predicted next character.
    """
    max = (float('-inf'), '')
    for char in vocabulary:
        newProb = calculate_probability(sequence, char, tables) > max[0]
        if newProb > max[0]:
            max = (newProb, char)
    return max[1]
