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
        # Predict the most likely next character
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char
        print(f"Updated sequence: {current_sequence}")

if __name__ == "__main__":
    # tables = create_frequency_tables("test.txt", 4)

    # print("Tables:", tables)
    
    # bestOption = predict_next_char("aa", tables, ('c', 'b', 'a'))
    # print(bestOption)

    #main()
    
    #print(predict_next_char('a', create_frequency_tables("Alice's Adventures in Wonderland.txt", 100), ('t', 'l', 'n')))
    print(predict_next_char('a', create_frequency_tables("warandpeace.txt", 20), ('t', 'l', 'n')))
