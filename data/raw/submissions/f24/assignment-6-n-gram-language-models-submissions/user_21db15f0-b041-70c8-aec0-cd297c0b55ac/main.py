from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    # document = read_file('warandpeace.txt')
    document = read_file("Alice's Adventures in Wonderland.txt")
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)
    # print_table(tables[0], 1)
    
    # print(tables[0])
    vocabulary = set(tables[0])
    # print("debug: vocabulary is " + str(vocabulary))
    
    current_sequence = initial_sequence
    # print(calculate_probability(initial_sequence, tables=tables))

    for _ in range(k):
        # Predict the most likely next character
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")
        # I added this line because without it the predictions got unecessarily long.
        if len(current_sequence) > len(initial_sequence) + k:
            break

if __name__ == "__main__":
    main()
