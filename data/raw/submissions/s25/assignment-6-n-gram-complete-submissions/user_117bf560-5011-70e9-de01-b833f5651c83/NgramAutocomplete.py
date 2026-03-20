from fractions import Fraction
import itertools

##Helper method used to generate list of every possible string from an vocabulary
def generate_strings(vocabulary, length):
    return [''.join(p) for p in itertools.product(vocabulary, repeat=length)]

##Helper method to convert a string to a cooresponding frequnecy
def string_to_frequency(string): 
    freq = "f(" + string[0] + "|"
    freq += ','.join(string[1:]) + ")"  # Handles all but the first character in one join
    return freq

def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    ##Finds vocabulary of a document - the characters used in the document
    vocabulary = sorted(set(document))
    frequencyTables = []
    
    for x in range(1, n+1):
        table = {}
        ##This code deals with finding the frequencies for sequences greater than 1.
        if x != 1:
           ##This code  generates every possible string from the vocabulary of length x, and creates the 
           ##cooresponding frequency for it in the table.
            strings = generate_strings(vocabulary, x)
            strings.sort()
            for string in strings:
                freq = string_to_frequency(string)
                table[freq] = 0 ##Appends frequency to table with initial value 0
            ##This code counts the actua frequencies
            for i in range(len(document) - x + 1):  
                gram = string_to_frequency(document[i:i+x])
                table[gram] += 1
        else: ##deals with individual character frequencies
            for char in vocabulary:
               table["f(" + char + ")"] = 0
            for i in range(len(document) - x + 1):  
                gram = "f(" + document[i] +")"
                table[gram] += 1
        frequencyTables.append(table)
    return frequencyTables

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
    length = len(sequence) + 1
    completeGram = sequence + char
    probability = 1
    for i in range(0, length):
        if i == 0 or n== 1: ##If the first character
            key = "f(" + completeGram[0] + ")" 
            probability *= tables[0].get(key, 0)/sum(tables[0].values()) ##frequency of first character / length of vocabulary
        elif i >= n: ## If i is greater than or equal to the max table size
            if i - n < 0 or i - 1 < 0:
                print(f"Index out of bounds for {i}")
                return 0
            key = string_to_frequency(completeGram[i - n + 1 : i+1]) 
            prev_key = string_to_frequency(completeGram[i-n + 1: i])
            probability *= tables[n-1].get(key, 0)/tables[n-2].get(prev_key, 1)##freq of next char w.r.t. n previous chars/ ##freq of prev char w.r.t. n-1 previous chars
        elif i == 1: ##If the second character
            key = string_to_frequency(completeGram[0: i+1])
            prev_key = "f(" + completeGram[0] + ")"
            probability *= tables[1].get(key, 0)/tables[0].get(prev_key, 1) ##frequency of second character w.r.t. first character/ frequency of first char
        else: ##For all other characters
            key = string_to_frequency(completeGram[0: i+1])
            prev_key = string_to_frequency(completeGram[0: i])
            probability *= tables[i].get(key, 0)/tables[i-1].get(prev_key, 1) ##frequency of next char w.r.t. the previous chars/ freq of prev char w.r.t. it's previous chars
    return probability
    

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
    probabilities = []
    for char in vocabulary:
        probabilities.append({'char': char, 'probability':calculate_probability(sequence, char[2], tables)})
    return max(probabilities, key=lambda x: x['probability'])['char'][2]
