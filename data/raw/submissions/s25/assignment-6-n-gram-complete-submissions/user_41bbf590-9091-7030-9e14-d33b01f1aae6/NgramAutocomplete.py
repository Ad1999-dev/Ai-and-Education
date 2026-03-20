def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    table_list = []
    #create each table
    for i in range(1, n+1):
        table = {}

        #for each contiguous sequence of i letters in the document (j=end index of sequence)
        for j in range(i, len(document)+1):
            seq = document[j-i:j]
            char = seq[-1] #last char in seq
            prev_chars = seq[0:-1] #everything before last char

            #if char doesn't exist as a key in table, add it
            if char not in table:
                table[char] = {}
            #if prev_chars doesn't exist as a key in char, add it
            if prev_chars not in table[char]:
                table[char][prev_chars] = 0

            #increment count - # times char appears after prev_chars
            table[char][prev_chars] += 1

        table_list.append(table)

    return table_list

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
    n = len(tables)
    seq = sequence + char
    product = 1

    #calculate each conditional probability for the sequences, multiply to product
    for i in range(1, len(seq)+1):
        #max length of "window" is n, so len(prev) <= n-1
        subseq = seq[max(0,i-n):i]
        subseqlen = len(subseq)
        #P(char|prev) = f(prev+char)/f(prev)
        #prev=prevprev+prevchar
        #f(prev+char)/f(prev) = f(char|prev)/f(prevchar|prevprev)
        char = subseq[-1] #last char in subseq
        prev = subseq[0:-1] #everything before last char

        #nonexistant sequence - probability = 0
        if (char not in tables[subseqlen-1]) or (prev not in tables[subseqlen-1][char]):
            return 0

        if i == 1: #special case: prev is empty (subseq = first char in seq)
            #sum all freqs in 1st table (size(document))
            sum = 0
            for letter in tables[0]:
                sum += tables[0][letter][""]
            #multiply f(char) / size(document) to product
            product *= tables[0][char][""] / sum

        else: #prev has len at least 1
            prevchar = prev[-1] #last char in prev
            prevprev = prev[0:-1] #everything before last char in prev

            #checking for /0
            if (prevchar not in tables[subseqlen-2]) or (prevprev not in tables[subseqlen-2][prevchar]):
                return 0

            #multiply f(char|prev) / f(prevchar|prevprev) to product
            product *= tables[subseqlen-1][char][prev] / tables[subseqlen-2][prevchar][prevprev]

    return product

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
    probs = {}
    #calculate prob for each next char
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        probs[char] = prob
    #find max and return key (char)
    return max(probs, key=probs.get)

#Test code
#tables = create_frequency_tables("aababcaccaaacbaabcaa", 3)
#print(tables)
#print(calculate_probability("aa", "c", tables))
#print(predict_next_char("", tables, set("abc")))