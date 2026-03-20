def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    # identify all unique characters in the document
    distChars = set()
    for char in document:
        distChars.add(char)

    # create an array of dicts with all possible sequences initially 0
    freqTables = []
    for gram in range(1, n+1):
        freqTable = {}

        # define a function to recursively add all possible char sequences of gram length
        def createStrings(curr=""):
            if (len(curr) == gram):
                freqTable[curr] = 0
                return
            # try curr + all possible chars
            for char in distChars:
                createStrings(curr + char)
        createStrings()
        freqTables.append(freqTable)

    # define function that updates string frequencies for a given word and gram
    def updateFreqs(gram, document):
        # gets starting indexes for all possible substrings of length gram in word
        for i in range(len(document) - gram + 1):
            freqTables[gram - 1][document[i:i+gram]] += 1

    # update frequencies according to the input document
    for g in range(1, n+1):
        updateFreqs(g, document)

    return freqTables


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
    # get the length of the input array 'tables'
    n = len(tables)

    fullSequence = sequence + char

    # initialize the probability as 1(we will multiply p repeatedly to get a joint probability)
    p = 1

    # define function to calculate the ith term of the joint probability
    def termJP(i):
        # case when calculating first term => need to calculate total letters in 'tables'
        if (i == 1):
            totalLetters = 0
            for letter in tables[0]:
                totalLetters += tables[0][letter]
            return tables[0][fullSequence[0]] / totalLetters

        # case when there are at least i tables in 'tables'
        elif (i <= n):
            # in this scenario, we can compute this term of the calculation without approximation
            return tables[i-1][fullSequence[:i]] / tables[i-2][fullSequence[:i-1]]
        
        # case when there are less than i tables in 'tables'
        else:
            # in this scenario, we must approximate this term by calculating a probability using only the last n terms of the sequence ending at letter i
            if (n == 1):
                totalLetters = 0
                for letter in tables[0]:
                    totalLetters += tables[0][letter]
                return tables[n-1][fullSequence[i-n:i]] / totalLetters
            return tables[n-1][fullSequence[i-n:i]] / tables[n-2][fullSequence[i-n:i-1]]

    # multiply p by each of the i terms in the joint probability sequence of probabilities
    for i in range(1, len(fullSequence)+1):
        if (p == 0):
            return 0
        p *= termJP(i)

    return p


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
    pred = ' '
    predP = 0

    for char in vocabulary:
        currP = calculate_probability(sequence, char, tables)
        if currP >= predP:
            predP = currP
            pred = char
    
    return pred
