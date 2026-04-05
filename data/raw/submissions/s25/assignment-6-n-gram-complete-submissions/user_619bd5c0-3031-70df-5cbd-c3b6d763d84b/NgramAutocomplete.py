from collections import defaultdict

def create_frequency_tables(document, n):
    tables = [defaultdict(int) for _ in range(n)]
    
    for i in range(len(document)):
        for k in range(min(n, len(document) - i)):
            ngram = document[i:i+k+1]
            tables[k][ngram] += 1
                
    return tables

def calculate_probability(sequence, char, tables):
    full_sequence = sequence + char
    prob = 1.0
    for i in range(len(full_sequence)):
        k = min(i, len(tables)-1)  # n-1 context
        context = full_sequence[max(0,i-k):i]
        ngram = context + full_sequence[i]
        
        if i == 0:  # P(x₁)
            prob *= tables[0][ngram] / sum(tables[0].values())
        else:  # P(xᵢ|xᵢ₋₁...)
            prob *= tables[k][ngram] / tables[k-1].get(context, 1)
    return prob

def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character using the frequency tables.
    """
    max_prob = -1
    best_char = None
    
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > max_prob:
            max_prob = prob
            best_char = char
            
    return best_char if best_char is not None else list(vocabulary)[0]