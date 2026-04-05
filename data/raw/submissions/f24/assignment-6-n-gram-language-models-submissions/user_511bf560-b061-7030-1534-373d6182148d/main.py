from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = read_file('warandpeace.txt')
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)

    vocabulary = set(tables[0])
    
    current_sequence = initial_sequence

    for _ in range(k):
        ontext = current_sequence[-(n - 1):] if len(current_sequence) >= (n - 1) else current_sequence
        # Predict the most likely next character
        next_char = predict_next_char(context, tables, vocabulary)
        print(f"Predicted next character: '{next_char}' based on context: '{context}'")
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")

if __name__ == "__main__":
    main()
