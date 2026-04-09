from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = read_file('warandpeace.txt')
    #document = 'aababcaccaaacbaabcaa'
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)

    
    for i, table in enumerate(tables):
        print(f"\nFrequency Table for {i + 1}-grams:")
        for seq, freq in sorted(table.items(), key=lambda x: (-x[1], x[0]))[:26]:  # Top 26 most frequent
            print(f"  {repr(seq)}: {freq}")
    
    
    """
    test_seq = input("Enter a sequence to test (e.g. 'the'): ")
    test_char = input("Enter the next character to predict probability for (e.g. 'r'): ")

    prob = calculate_probability(test_seq, test_char, tables)
    print(f"Probability of '{test_char}' given '{test_seq}' is {prob}")
    """


    vocabulary = set(tables[0])
    
    current_sequence = initial_sequence

    for _ in range(k):
        # Predict the most likely next character
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")

if __name__ == "__main__":
    main()
