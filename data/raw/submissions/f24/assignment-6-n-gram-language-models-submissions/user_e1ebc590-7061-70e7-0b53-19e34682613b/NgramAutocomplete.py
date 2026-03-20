from collections import defaultdict
import random


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
    for i in range(1,n+1):
        table_i = defaultdict(float)
        for j in range(0, len(document) - i + 1):
            table_i[document[j:j+i]] += 1
        tables.append(table_i)
    return tables

#alternate version
def calculate_probability_alt(sequence, char, tables):
    n = len(sequence)

    #constrain sequence length to n
    if n + 1> len(tables):
        sequence = sequence[n + 1 -len(tables):]
        print(sequence)

    #conditional probability: P(X_i = char | sequence) = P(sequence + char)/P(sequence)
    P = tables[n][sequence + char] / tables[n-1][sequence] if tables[n-1][sequence] > 0 else 0.0
    return P

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
    sequence = sequence + char

    n = len(tables)
    P = 1
    left = 0
    for right in range(0, len(sequence)):
        if (right - left) >= n:
            left += 1
        top_seq = sequence[left:right+1]
        top_freq = tables[right-left][top_seq]
        # print("top", top_seq)

        if (right - left) >= 1:
            bot_seq = sequence[left:right]
            bot_freq = tables[right-left-1][bot_seq]
            # print("bot", bot_seq)
        else:
            bot_freq = sum(tables[0].values()) #document 
        # print("ratio: ",top_freq,"/",bot_freq)

        #bottom sequence doesn't exist -> P(sequence) = 0
        if bot_freq == 0:
            return 0
        
        P *= top_freq/bot_freq #P(subsequence + next char)/P(subsequence)
    return P


# table = create_frequency_tables("aababcaccaaacbaabcaa", 3)
# P = calculate_probability("aababc","a", table)
# print(P)

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
    default_chars = "qwertyuiopasdfghjklzxcvbnm"
    best_char = random.choice(default_chars)
    best_prob = 0.0
    for c in vocabulary:
        P = calculate_probability(sequence, c, tables)
        if P > best_prob:
            best_prob = P
            best_char = c
    return best_char

