from collections import defaultdict, deque

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The maximum size of n-grams to track.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    result = [defaultdict(int) for _ in range(n)]
    
    with open(document, 'r', encoding='utf-8') as file:
        window = deque(maxlen=n)

        for line in file:
            for char in line:
                window.append(char)
                
                for i in range(1, n+1):
                    if len(window) >= i:
                        ngram = tuple(window)[-i:]
                        result[i-1][ngram] += 1
    
    return result

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
    before = list(sequence)
    before.append(char)
    seqList = tuple(before)
    
    #obtain the frequency of the target
    probability = tables[len(before) - 1][seqList]
    
    total = 0
    
    #find total in n = 1
    for num in tables[0].values():
        total += num

    #divide by the total of the total from table n = 1
    return probability / total


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
    
    max = [0, '']
    
    for char in vocabulary:
        current = calculate_probability(sequence, char, tables)
        if current > max[0]:
            max = [current, char]

    return max[1]
