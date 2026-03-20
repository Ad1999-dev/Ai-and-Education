from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = read_file('warandpeace.txt')
    # n, initial_sequence, k, c = 3, 'a', 1, 'b'
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    c = input("Enter the character following the sequence (c): ")
    
    tables = create_frequency_tables(document, n)
    # for table in tables:
        # print(table)
    
    prob = calculate_probability(initial_sequence, c, tables)
    # print(prob)

    vocabulary = set(tables[0])
    # print("*" * 50)
    # print(vocabulary)
    
    current_sequence = initial_sequence

    for _ in range(k):
        # Predict the most likely next character
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char      
        # print(f"Updated sequence: {current_sequence}"
        # if next_char != 'a':
            # print("NOT CRAZY")

if __name__ == "__main__":
    main()
    print("Done.")
