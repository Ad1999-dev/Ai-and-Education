import re
import numpy as np
def create_frequency_tables(document, N):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    s = re.sub('[\"?(#$%&:;*•™)\\s!\n _]+', ' ', document)
    dict_arr = []

    for n in range(1, N+1):
        char_dict = {}
        for i in range(len(s)-n+1):
            char = s[i:i+n]
            if char in char_dict:
                char_dict[char] += 1
            else:
                char_dict[s[i:i+n]] = 1
        dict_arr.append(char_dict)
    for n in range(N - 1, -1, -1):
        char_dict = dict_arr[n]
        for k in char_dict.keys():
            if n != 0:
                prev = k[:-1]
                char_dict[k] = char_dict[k] / dict_arr[n-1][prev]
            else:
                char_dict[k] = char_dict[k] / len(s)
    return dict_arr


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
    new_seq = sequence + char
    length = len(new_seq)
    prob = tables[length-1][new_seq] if new_seq in tables[length-1] else 0
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
    sequence = sequence.lower()
    full_seqs = [sequence + vocab if len(sequence) < len(tables) else sequence[-len(tables)+1:] + vocab for vocab in vocabulary]
    seq_length = len(sequence) if len(sequence) < len(tables) else len(tables) - 1
    next_pos = np.argmax([tables[seq_length][seq] if seq in tables[seq_length] else 0 for seq in full_seqs])
    return full_seqs[next_pos][-1]