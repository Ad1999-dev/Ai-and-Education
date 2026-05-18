import collections
from typing import List
from utilities import print_table


def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    tables = [collections.Counter() for _ in range(n)]
    L = len(document)
    for k in range(1, n+1):
        # for each valid start position of a k‑gram...
        for start in range(L - k + 1):
            gram = document[start:start+k]
            tables[k-1][gram] += 1
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
    #print(tables)
    n = len(tables)
    k = len(sequence)
    conLen = min(k, n-1)

    if conLen == 0:
        num = tables[0].get(char,0)
        denom = sum(tables[0].values())
    else:
        ctx   = sequence[-conLen:]                     # last `order` chars
        num   = tables[conLen].get(ctx + char, 0)      # count of (order+1)-gram
        denom = tables[conLen-1].get(ctx, 0)           # count of order-gram

    return (num/denom) if denom > 0 else 0.0




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
    maxChar = None
    max = -1
    for i in vocabulary: 
        prob = calculate_probability(sequence, i, tables)
        if prob > max: 
            max = prob
            maxChar = i
    return maxChar if maxChar is not None else vocabulary.get(0)

if __name__ == "__main__":
    # Load your document, for example:
    #document = "this is a sample document for testing n-gram frequency analysis"
    document = "aababcaccaaacbaabcaa"
    #document = "aaabababbbbabbbbbbabbabaa"
    n = 3
    frequency_tables = create_frequency_tables(document, n)
    #print(frequency_tables)
    # Print the frequency tables
    #for i, table in enumerate(frequency_tables, start=1):
        #print(f"Frequency Table for n={i}:")
        #for context, freq in table.items():
            #print(f"  '{context}': {freq}")

    #defghijklmnopqrstuvwxyz
    alphabet = "abc"
    se = ""
    alphTab = [alphabet[i] for i in range(len(alphabet))]
    #str = calculate_probability(se, "a", frequency_tables)
    #print(str)
    print(predict_next_char("", frequency_tables, alphTab))
