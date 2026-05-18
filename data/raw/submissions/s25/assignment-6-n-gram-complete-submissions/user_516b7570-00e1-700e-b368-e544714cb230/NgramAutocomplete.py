import time
def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    ngram = {}
    num = []
    for j in range(1,n+1):
        for i in range(len(document)-j+1):
            if document[i] not in num: # Getting a list of all characters in document
                num.append(document[i])
            temp = document[i:i+j]
            if temp in ngram:
                ngram[temp] += 1
            else:
                ngram[temp] = 1
    ngram['total_len'] = len(document)
    ngram['n_val'] = n # The k in the k-gram, I didn't use it but could be useful
    ngram['letters'] = num
    #print(ngram)
    return ngram


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

    if sequence[0] not in tables: return 0
    prob = tables[sequence[0]] / tables['total_len']

    for i in range(1,len(sequence)):
        if sequence[:i+1] not in tables: return 0

        prob *= ((tables[sequence[:i+1]] / tables[sequence[:i]]))

    if sequence+char not in tables: return 0
    prob *= tables[sequence+char] / tables[sequence]
    return prob

    # If statements are to just check if the word is in tables, return 0 if not


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
    # Loop through each charecter, putting each charecter into a function
    chars = tables['letters']
    max_prob = 0
    letter = "" # Returns nothing at the end if no possible sequence was detected
    #listo = []
    perplist = []
    for char in chars:
        prob = calculate_probability(sequence,char,tables)
        if prob == 0: continue
        perplist.append([char, perplexity(prob,sequence+char)])
        #listo.append([prob,char])
        if prob > max_prob:
            letter = char
            max_prob = prob
    #listo.sort()
    #print("Char: " + str(i[1]) + " Prob: " + str(i[0]))
    #perplist2 = sorted(perplist, key=lambda x: x[1])
    #for i in perplist2[:3]:
    #    print(i)
    return letter

# Commented out, but the code is right above commented out
def perplexity(prob, sequence):
    if prob == 0: return 69
    perp = (1 / prob) ** (1/len(sequence))
    return perp
