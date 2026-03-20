from utilities import print_table
from collections import defaultdict

def create_frequency_tables(document, n):
    tables = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
    
    for i in range(len(document)):
        for k in range(1, n + 1): 
            if i - k < 0:
                continue
            context = document[i - k:i]
            char = document[i]
            tables[k - 1][context][char] += 1

    return tables


def calculate_probability(sequence, char, tables):
    n = len(sequence)
    if n == 0:
        table = tables[0]
        total_count = sum(table.values())
        char_count = table[char]
    else:
        table = tables[n - 1]
        context_dict = table.get(sequence, {})
        
        total_count = sum(context_dict.values())
        char_count = context_dict.get(char, 0) 

    return char_count / total_count if total_count > 0 else 0




def predict_next_char(sequence, tables, vocabulary):
    max_prob = 0
    best_char = None
    
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char
    
    return best_char if best_char else ' '
