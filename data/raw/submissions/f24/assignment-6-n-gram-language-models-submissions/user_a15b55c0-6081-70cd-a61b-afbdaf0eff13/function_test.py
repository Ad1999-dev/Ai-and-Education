from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

tables = create_frequency_tables('aababcaccaaacbaabcaa', 3)

print_table(tables, 2)

probabilities = {}


for char in ('a','b','c'):
    prob = calculate_probability('c', char, tables)
    probabilities[char] = prob

print(predict_next_char('a', tables, ('a','b','c')))
#print(probabilities)