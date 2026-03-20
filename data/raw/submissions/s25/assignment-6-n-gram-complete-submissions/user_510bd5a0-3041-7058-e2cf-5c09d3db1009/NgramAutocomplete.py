from collections import defaultdict

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
    
    # Create each table (1-gram to n-gram)
    for i in range(1, n + 1):
        # For each i-gram, create a frequency table
        table = defaultdict(lambda: defaultdict(int))
        
        # Slide through the document with a window of size i
        for j in range(len(document) - i + 1):
            ngram = document[j:j+i]
            
            # The current character is the last one in the n-gram
            current_char = ngram[-1]
            
            # Context is all the preceding characters in the n-gram
            context = ngram[:-1]
            
            # increment the count for this character in this context
            table[current_char][context] += 1
        
        # Convert defaultdicts to regular dicts 
        fixed_table = {}
        for char, contexts in table.items():
            fixed_table[char] = dict(contexts)
        
        tables.append(fixed_table)
    
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
    n = len(tables)
    
    # Ensure we only use at most n-1 characters from the sequence
    if len(sequence) >= n:
        sequence = sequence[-(n-1):]
    
    
    table_index = len(sequence)
    
    # Ensure we have a valid table index
    if table_index >= n:
        table_index = n - 1
    
    
    table = tables[table_index]
    
    # Check if the character exists in our frequency table
    if char not in table:
        return 0.0
    
    # Check if the sequence exists as a context for this character
    if sequence not in table[char]:
        return 0.0
    
    # Calculate the total occurrence count for this context
    context_total = 0
    for c in table:
        if sequence in table[c]:
            context_total += table[c][sequence]
    
    # Return the conditional probability
    if context_total > 0:
        return table[char][sequence] / context_total
    else:
        return 0.0
   
    
    
    


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
    n = len(tables)
    
    # Use only the last n-1 characters as context
    context = sequence
    if len(sequence) > n-1:
        context = sequence[-(n-1):]
    
    # Calculate probability for each character in the vocabulary
    best_char = None
    best_prob = -1
    
    for char in vocabulary:
        # Get probability for this character following the context
        prob = calculate_probability(context, char, tables)
        
        # Update best character if this one has higher probability
        if prob > best_prob:
            best_prob = prob
            best_char = char
    
    # If we found a character with non-zero probability, return it
    if best_char is not None and best_prob > 0:
        return best_char
    
    # If we couldn't find a prediction using the full context, try with a shorter context
    if len(context) > 0:
        shorter_context = context[1:]
        return predict_next_char(shorter_context, tables, vocabulary)
    
    # Last Last Lastest resort
    # Find the most common character in the unigram model
    if tables and vocabulary:
        unigram_table = tables[0]
        most_common_char = None
        max_count = -1
        
        for char in unigram_table:
            if "" in unigram_table[char]:
                count = unigram_table[char][""]
                if count > max_count:
                    max_count = count
                    most_common_char = char
        
        if most_common_char is not None:
            return most_common_char
    
    

    

  
    
    
