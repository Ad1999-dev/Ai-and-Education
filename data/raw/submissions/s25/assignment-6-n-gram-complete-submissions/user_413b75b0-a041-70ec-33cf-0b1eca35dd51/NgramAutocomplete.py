from collections import Counter, defaultdict
from typing import List, Dict, Union

def _update(tables, char, context):
    i = len(context)                    
    tables[i][char][context] += 1

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    tables = [defaultdict(Counter) for _ in range(n)]

    doc_len = len(document)
    for idx, char in enumerate(document):
        for i in range(n):                    
            if idx - i < 0:
                break                           
            context = document[idx - i: idx]     
            _update(tables, char, context)
    return tables

def _context_probability(tables, context, char):
    for i in range(len(context), -1, -1):           
        sub_ctx = context[-i:]                     
        table     = tables[i]                   
        char_cnt  = table[char].get(sub_ctx, 0)
        if i == 0:
            denom = sum(c[''].values() for c in tables[0].values())  
        else:
            denom = sum(t.get(sub_ctx, 0) for t in table.values())

        if denom:                                
            return char_cnt / denom
    return 0.0


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

    max_ctx_len = len(tables) - 1
    context = sequence[-max_ctx_len:]             
    return _context_probability(tables, context, char)


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

    probs = {c: calculate_probability(sequence, c, tables) for c in vocabulary}
    if all(p == 0 for p in probs.values()):
        unigram_counts = {c: tables[0][c][''] for c in vocabulary}
        return max(unigram_counts, key=unigram_counts.get)

    return max(probs, key=probs.get)
