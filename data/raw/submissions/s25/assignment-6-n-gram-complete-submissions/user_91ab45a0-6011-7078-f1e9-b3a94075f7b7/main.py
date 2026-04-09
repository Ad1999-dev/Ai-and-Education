from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = read_file("warandpeace.txt")
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

    # chars = ['a','b','c']
    # document = "aababcaccaaacbaabcaa"
    # tables = create_frequency_tables(document, 3)
    # print(tables[0]['a'][''])
    # print_table(tables, 4)

    # for i in range(3):
    #     for char in tables[i].keys():
    #         for prev_seq in tables[i][char].keys():
    #             print(str(calculate_probability(prev_seq, char, tables)) + ", ")
    #     print()

    # print(str(calculate_probability("c",'c', tables)))

    # print(str(calculate_probability("aa",'c', tables)))
    # print(str(calculate_probability("ab",'c', tables)))
    # print(str(calculate_probability("ac",'c', tables)))
    # print(str(calculate_probability("ba",'c', tables)))
    # print(str(calculate_probability("bb",'c', tables)))
    # print(str(calculate_probability("bc",'c', tables)))
    # print(str(calculate_probability("ca",'c', tables)))
    # print(str(calculate_probability("cb",'c', tables)))
    # print(str(calculate_probability("cc",'c', tables)))

    # print(predict_next_char("aa", tables, chars))

if __name__ == "__main__":
    main()
