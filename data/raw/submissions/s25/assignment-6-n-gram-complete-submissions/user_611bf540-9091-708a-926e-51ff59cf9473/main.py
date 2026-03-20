from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = read_file('warandpeace.txt')
    #document = read_file('AlicesAdventuresinWonderland.txt')
    #document = read_file('test.txt')
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)
    #3print_table(tables, n)

    vocabulary = set(tables[0])
    
    current_sequence = initial_sequence
    #print(tables)
    for _ in range(k):
        # Predict the most likely next character
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")
    
    
    ##Testing only
    # char = input("Enter a char: ")
    # prob = calculate_probability(initial_sequence, char, tables)
    # print(prob)

if __name__ == "__main__":
    main()
