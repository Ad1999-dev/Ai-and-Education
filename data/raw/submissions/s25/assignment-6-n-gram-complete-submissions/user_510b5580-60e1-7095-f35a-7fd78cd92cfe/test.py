
def make_tables(tables):

    for i in range(0, n):
        table_n = {}

        for j, char in enumerate(text):
            
            if (j < len(text)-(i)):
                k = text[j:j+(i+1)]

                if k in table_n:
                    table_n[k] += 1
                        
                else:
                    table_n[k] = 1
        tables[i] = table_n
    return tables

def calc_prob(sequence, char, tables):
    n = len(tables)

    sequence = sequence+char
    product = 1

    size_c = 0
    for i in tables[0]:
        size_c+= tables[0][i]

    for i, char in enumerate(sequence):
        context_size = min(i, n-1)
        context = sequence[i-context_size:i+1]
        
        if context in tables[context_size]:
            numer = tables[context_size][context]
        else:
            numer = 0
        
        if context_size == 0:
            denom = size_c
        else:
            denom = tables[context_size-1][sequence[i-context_size:i]]
        
        product *= numer/denom
    
    return product

def predict_next_char(sequence, tables, vocabulary):
    next_char = ''
    next_char_p = 0

    for i in vocabulary:
        char_p = calc_prob(sequence, i, tables)

        if char_p > next_char_p:
            next_char = i
            next_char_p = char_p

    return next_char

tables = {}

vocabulary = ['a', 'b', 'c']

n = 3

text = "aababcaccaaacbaabcaa"

sequence = "ccc"
char = "b"

tables = make_tables(tables)

next_char = predict_next_char(sequence, tables, vocabulary)

print(next_char)