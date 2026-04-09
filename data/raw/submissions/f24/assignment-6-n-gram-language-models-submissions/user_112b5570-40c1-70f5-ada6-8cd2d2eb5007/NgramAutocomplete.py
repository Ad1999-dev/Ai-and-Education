from pprint import pprint
def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    # This is a more concrete table, we're writing what occurences we see
    frequency_table = [{} for _ in range(n)] #create list of n dictionaries

    for i in range(len(document)): # iterating over document
        for size in range(1, n+1): # adding to each n-gram
            if i + size <= len(document): # in case we go out of bounds
                substring = tuple(document[i:size+i]) # get the substring

                #increment if exists, initialize if not
                if tuple(substring) in frequency_table[size - 1]:
                    frequency_table[size-1][substring] += 1 
                else:
                    frequency_table[size-1][substring] = 1 
    return frequency_table


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.

    - **Returns**:
        - Returns a probability value for the sequence.
    """

    seq_length = len(sequence)
    
    if seq_length < len(tables):
        conditional_ngram = tuple(sequence + char)
        context_ngram = tuple(sequence)
        
        conditional_count = tables[seq_length].get(conditional_ngram, 0) # get probability of conditional, default 0
        context_count = tables[seq_length - 1].get(context_ngram, 0)  # get probability of context, default 0
        
        if context_count > 0:
            return conditional_count / context_count
        else:
            return 0
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

    probabilities = {}

    for char in vocabulary:
        probabilities[char] = calculate_probability(sequence, char, tables)

    most_likely_char = max(probabilities, key=probabilities.get)



    return most_likely_char

# Testing
test_table = create_frequency_tables("aababcaccaaacbaabcaa", 3)
'''
print(calculate_probability("aa", "a", test_table))
print(calculate_probability("aa", "b", test_table))
print(calculate_probability("aa", "c", test_table))
print(calculate_probability("ab", "a", test_table))
print(calculate_probability("ab", "b", test_table))
print(calculate_probability("ab", "c", test_table))
print(calculate_probability("ac", "a", test_table))
print(calculate_probability("ac", "b", test_table))
print(calculate_probability("ac", "c", test_table))

print(calculate_probability("ba", "a", test_table))
print(calculate_probability("ba", "b", test_table))
print(calculate_probability("ba", "c", test_table))
print(calculate_probability("bb", "a", test_table))
print(calculate_probability("bb", "b", test_table))
print(calculate_probability("bb", "c", test_table))
print(calculate_probability("bc", "a", test_table))
print(calculate_probability("bc", "b", test_table))
print(calculate_probability("bc", "c", test_table))

print(calculate_probability("ca", "a", test_table))
print(calculate_probability("ca", "b", test_table))
print(calculate_probability("ca", "c", test_table))
print(calculate_probability("cb", "a", test_table))
print(calculate_probability("cb", "b", test_table))
print(calculate_probability("cb", "c", test_table))
print(calculate_probability("cc", "a", test_table))
print(calculate_probability("cc", "b", test_table))
print(calculate_probability("cc", "c", test_table))
'''
print(predict_next_char("aa", test_table, "abc"))