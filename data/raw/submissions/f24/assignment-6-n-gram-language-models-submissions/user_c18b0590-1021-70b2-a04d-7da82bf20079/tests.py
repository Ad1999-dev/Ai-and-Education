from NgramAutocomplete import calculate_probability, create_frequency_tables


document = "aababcaccaaacbaabcaa"
tables = create_frequency_tables(document, 3)

# Example Test
sequence = "aa"
char = "a"
probability = calculate_probability(sequence, char, tables)
print(f"P({char} | {sequence}) = {probability}")