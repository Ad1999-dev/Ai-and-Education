from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = read_file("Alice's Adventures in Wonderland.txt")
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)

    vocabulary = set(tables[0].keys())  # Unigrams as base vocabulary
    for table in tables[1:]:  # For higher-order n-grams (bigrams, trigrams, etc.)
        for char in table.keys():
            vocabulary.add(char)
    
    current_sequence = initial_sequence

    for _ in range(k):
        # Predict the most likely next character
        context = current_sequence[-n-1:] if n > 1 else ""
        next_char = predict_next_char(context, tables, vocabulary)
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")

if __name__ == "__main__":
    main()
