from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = read_file('Alice.txt')  # Read the training document
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    # Create frequency tables for n-grams
    tables = create_frequency_tables(document, n)

    # Extract the vocabulary from the unigram table
    vocabulary = set(tables[0])
    
    # Initialize the sequence with the user-provided input
    current_sequence = initial_sequence

    for _ in range(k):
        # Slice the context to n-1 characters for prediction
        context = current_sequence[-(n-1):] if len(current_sequence) >= n-1 else current_sequence
        next_char = predict_next_char(context, tables, vocabulary)  # Predict the next character
        current_sequence += next_char  # Append the predicted character to the sequence
        print(f"Updated sequence: {current_sequence}")  # Display the updated sequence

if __name__ == "__main__":
    main()

