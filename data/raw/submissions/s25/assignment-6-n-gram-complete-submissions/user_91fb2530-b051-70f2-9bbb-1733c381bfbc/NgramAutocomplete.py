from collections import defaultdict
import re

def create_frequency_tables(document, n):
    """
    Constructs a list of n frequency tables for an n-gram model.
    Each table[i] represents an (i+1)-gram model where:
    - Key is a character
    - Value is a dictionary mapping context (previous chars) to frequency
    """
    clean_document = re.sub(r'[^a-z\s\.,!?\-\']', '', document.lower())
    
    tables = []
    for i in range(1, n+1):
        table = defaultdict(lambda: defaultdict(int))
        for idx in range(len(clean_document) - i + 1):
            context = clean_document[idx:idx+i-1] if i > 1 else ''
            char = clean_document[idx+i-1]
            table[char][context] += 1
        tables.append(table)
    return tables

def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing `char` after `sequence` using backoff.
    If we have data for the highest order n-gram, use that. Otherwise back off to lower orders.
    Apply smoothing to avoid zero probabilities.
    """

    # o = order
    for o in range(len(tables)-1, -1, -1):
        context_len = o
        if context_len == 0:
            context = ''  
        else:
           
            context = sequence[-context_len:] if len(sequence) >= context_len else sequence
        
        table = tables[o]
        
        
        if context in table[char] and table[char][context] > 0:
            
            total_count = sum(table[c][context] for c in table if context in table[c])
            
            if total_count > 0:
                
                vocab_size = len(table)
                prob = (table[char][context] + 0.1) / (total_count + 0.1 * vocab_size)
                return prob
    
   
    return 1.0 / len(tables[0])


def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character based on the given sequence.
    Only considers alphabetic characters, spaces, and common punctuation.
    """
   
    filtered_vocab = [c for c in vocabulary if c.isalnum() or c.isspace() or c in ".,!?'-\""]
    
    if not filtered_vocab:
        filtered_vocab = vocabulary
    
    prob_char = None
    prob_max = -1
    
    for char in filtered_vocab:
        prob = calculate_probability(sequence, char, tables)
        if prob > prob_max:
            prob_max = prob
            prob_char = char
    
    return prob_char
