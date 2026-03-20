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
   frequency_tables = [{} for _ in range(n)]
  
   document = document.replace('\n', ' ').replace('\r', '')
  
   length = len(document)
   for i in range(length):
       for k in range(1, n + 1):
           if i + k <= length:
               ngram = document[i:i + k]


               if ngram in frequency_tables[k - 1]: 
                   frequency_tables[k - 1][ngram] += 1
               else:
                   frequency_tables[k - 1][ngram] = 1
           else:
               break


   return frequency_tables




def calculate_probability(sequence, char, tables):
   """
   Calculates the probability of observing a given sequence of characters using the frequency tables.


   - **Parameters**:
       - `sequence`: The sequence of characters whose probability we want to compute.
       - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
    """
   n = len(tables)
   max_context_length = n - 1


   context = sequence[-max_context_length:] if max_context_length > 0 else ''
   context_length = len(context)

   ngram = context + char
   ngram_length = len(ngram)


   table_index = ngram_length - 1  

   if table_index >= len(tables):
       table_index = len(tables) - 1
       ngram = ngram[-(table_index + 1):]
       context = ngram[:-1]
       context_length = len(context)


   ngram_table = tables[table_index]
   ngram_freq = ngram_table.get(ngram, 0)


   if context_length > 0:
       context_table = tables[context_length - 1]
       context_freq = context_table.get(context, 0)
   else:
       context_freq = sum(tables[0].values())

   if context_freq > 0:
       prob = ngram_freq / context_freq
   else:
       prob = 0 


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
   max_prob = 0.0
   next_char = None
   n = len(tables)
   max_context_length = n - 1

   context = sequence[-max_context_length:] if max_context_length > 0 else ''

   for char in vocabulary:
       prob = calculate_probability(context, char, tables)

       if prob > max_prob:
           max_prob = prob
           next_char = char


   if next_char is None:
       next_char = ''  


   return next_char