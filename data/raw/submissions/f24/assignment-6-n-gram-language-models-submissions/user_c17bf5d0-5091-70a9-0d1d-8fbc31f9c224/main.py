from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char
import re

def main():
    document = read_file('Alices_Adventures_in_Wonderland.txt')

    document = document.lower()  
    document = re.sub(r'[^a-z\s]', '', document) 
    document = re.sub(r'\s+', ' ', document).strip() 
    
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)
    vocabulary = set(tables[0])
    
    current_sequence = initial_sequence

    for _ in range(k):
        # Predict the most likely next character
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")

if __name__ == "__main__":
    main()