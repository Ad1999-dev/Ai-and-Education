from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    # document = read_file('warandpeace.txt')
    document = read_file('AlicesAdventuresinWonderland.txt')
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)

    vocabulary = set(tables[0])
    print(vocabulary)
    
    current_sequence = initial_sequence

    for _ in range(k):
        # Predict the most likely next character
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")

if __name__ == "__main__":
    main()
    document = "aababcaccaaacbaabcaa"
    n = 3
    sequence = "aa"
    # tables = create_frequency_tables(document, n)
    # print_table(tables, n)
    prob = []
    one = ["a", "b", "c"]
    two = ["aa", "ab", "ac", "ba", "bb", "bc", "ca", "cb", "cc"]
    # for char in one:
    #     for pre in two:
    #         prob.append(calculate_probability(pre, char, tables))
    # for char in one:
    #     prob.append(calculate_probability("a", char, tables))

    # prob = calculate_probability("aa", "a", tables)

    # next_char = predict_next_char(sequence, tables, ["a", "b", "c"])
    # print(prob)
    # print(next_char)
