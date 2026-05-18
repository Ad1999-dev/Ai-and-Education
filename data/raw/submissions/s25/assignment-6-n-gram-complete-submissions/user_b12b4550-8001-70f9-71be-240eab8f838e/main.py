from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = read_file('/Users/nguyennam/Desktop/assigment_6_fix/assigment 6/assignment-6-n-gram-complete-NamNguyen122/assignment-6-n-gram-complete-NamNguyen122/warandpeace.txt')
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))

    tables = create_frequency_tables(document, n)

    # FIXED vocabulary extraction:
    vocabulary = set(char for char_seq in tables[0] for char in char_seq)

    current_sequence = initial_sequence

    for _ in range(k):
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char
        print(f"Updated sequence: {current_sequence}")

if __name__ == "__main__":
    main()
