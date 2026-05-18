def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    #Create n tables and put them in an array to organize
    tables = []
    i = 0
    while i < n:
        tables.append({})
        i += 1

    #Track the previous letters in an array in order to look back for the n grams
    prev = []
    for letter in document:
        j = 0
        #Loop through each table
        while j < n:
            if(len(prev) >= j):
                #Create previous letter chain
                k = 0
                key = ""
                while(k < j):
                    key = prev[len(prev) - 1 - k] + key
                    k += 1
                key += letter
                # Add to according table entry
                if(key in tables[j]):
                    tables[j][key] += 1
                else:
                    tables[j][key] = 1
                j += 1
            else:
                break
        prev.append(letter)

    return tables


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters prior to char.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    #isolate longest table usable for this sequence
    gram = 0
    if(len(tables) > len(sequence)):
        gram = len(sequence)
    else:
        gram = len(tables) - 1
    
    #use said table to calculate probability
    #P(word | sequence) = P(Word and Sequence) / P(Sequence)

    word = sequence + char

    probWord = 1
    probSequence = 1
    for i in range(len(word)):
        if(gram == 0 or i == 0):
            denom = sum(tables[0].values())
            num = 0 if word[i:i+1] not in tables[0] else tables[0][word[i:i+1]]
        elif(i <= gram):
            denom = 1 if word[:i] not in tables[i-1] else tables[i-1][word[:i]]
            num = 0 if word[:i+1] not in tables[i] else tables[i][word[:i+1]]
        else:
            denom = 1 if word[i-gram:i] not in tables[gram-1] else tables[gram-1][word[i-gram:i]]
            num = 0 if word[i-gram:i+1] not in tables[gram] else tables[gram][word[i-gram:i+1]]
        if(denom == 0):
            currProb = 1
        currProb = num/denom
        if(i < len(word)-1):
            probSequence *= currProb
        probWord *= currProb

    prob = 0 if probSequence == 0 else probWord / probSequence

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
    char = ""
    prob = -1

    for letter in vocabulary:
        currProb = calculate_probability(sequence, letter, tables)
        if currProb > prob:
            char = letter
            prob = currProb
            
    return char