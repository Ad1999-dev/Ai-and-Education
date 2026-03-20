from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = "aababcaccaaacbaabcaa"
    n = 3
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)
    vocabulary = {char for char in set(document) if char.isalnum()} 
    current_sequence = initial_sequence

    for _ in range(k):
        current_sequence = current_sequence[-n:]
        next_char = predict_next_char(current_sequence, tables, vocabulary)
        current_sequence += next_char
        print(f"Updated sequence: {current_sequence}")


if __name__ == "__main__":
    main()
