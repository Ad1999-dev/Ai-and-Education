



def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """

    #Get sequence of letters starting at 1, then 2, then 3... to n
    #Based on n make a dictionary of these Strings
    #If they appear make new entry, if they are in add
    def get_n_gramTable(document,x):
        start = 0
        end = x
        frequencytable = {}
        while end<len(document)+1:
            result = document[start:end]
            if result in frequencytable.keys():
                frequencytable[result] +=1
            else:
                frequencytable[result] = 1
            start +=1
            end+=1
        return frequencytable
    
    toReturn = []
    if (n<1):
        return []
    

    for j in range(1,n+1):
        t = get_n_gramTable(str(document),j)
        toReturn.append(t)
    return toReturn


    


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
        - `Char`: The character whose probability of occurrence after the sequence is to be calculated. (Added with campuswire)

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    #Longer way to do it, but represents how its done with bayesian networks
    if (len(char)!=1):
         raise Exception("char is not one character")
         
    if (len(sequence)) <1:
        raise Exception("Invalid sequence")
    
    if (len(tables)>=len(sequence)+1):
        table1 = tables[len(sequence)-1]
        table2 =tables[len(sequence)]
        #print(tables[len(sequence)])

        if sequence not in table1.keys():
            return 0
        if sequence+char not in table2.keys():
            return 0
        currentprob = 1.0
        for i in range(0,len(sequence)):
            #print(i)
            table = tables[i]
        
            frequency = table[sequence[0:i+1]]
            if i==0:
               total = sum(table.values())
            else:
                tableS = tables[i-1]
                total = tableS[sequence[0:i]]
        
            currentprob = currentprob*(frequency/total)
        #print(currentprob)
        #print(table.values())
        #print(sum(table.values()))
        #print(frequency,currentprob)
        
    #     currentprob = currentprob*frequency
    #     #print(currentprob)
        table = tables[len(sequence)]
        frequency = table[sequence+char]
        tableS = tables[len(sequence)-1]
        total = tableS[sequence]
        return (currentprob*frequency/total)/currentprob,10
    else:
        # table1 = tables[len(sequence)-1]
        # table2 =tables[len(sequence)]
        # #print(tables[len(sequence)])

        # if sequence not in table1.keys():
        #     return 0
        # if sequence+char not in table2.keys():
        #     return 0
        
        
        currentprob = 1.0
        currentseq = ""
        sequenceWC = sequence+char
        for i in range(0,len(sequenceWC)):
            currentseq= currentseq + sequenceWC[i]
            
            if (i>=len(tables)):
                table = tables[len(tables)-1]
                n = len(tables)
            else:
                table = tables[i]
                n = i+1
            if i==0:
                total = sum(table.values())
            else:
                total = last
            sequencetocalc = currentseq[-n:]
            
            frequency = table[sequencetocalc]
            last = frequency
            
            currentprob = (currentprob*(frequency/total))/currentprob
            
        return currentprob

            
            
        #print(currentprob)
        #print(table.values())
        #print(sum(table.values()))
        #print(frequency,currentprob)
        
    #     currentprob = currentprob*frequency
    #     #print(currentprob)
        


   

    # if (len(tables)<len(sequence)+1):
    #      raise Exception("Depth of tables not given")
    
    # if (len(char)!=1):
    #      raise Exception("char is not one character")
         
    # if (len(sequence)) <1:
    #     raise Exception("Invalid sequence")
    
    # table = tables[len(sequence)-1]
    # if sequence not in table.keys():
    #     return 0
    # denom = table[sequence]
    # table2 = tables[len(sequence)]
    # if sequence+char not in table2.keys():
    #     return 0
    # numer = table2[sequence+char]
    # return numer/denom
    


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
    
        
    start = 0
    char = 0
    for i in vocabulary:
        x = calculate_probability(sequence,i,tables)
        if x > start:
            start = x
            char = i
    return char
