from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = read_file('warandpeace.txt')
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    # document = 'aababcaccaaacbaabcaa'
    # n = 3
    
    tables = create_frequency_tables(document, n)
    # for table in tables:
    #     sum_table = sum(table.values())
    #     print(table)
    #     print(f"Sum of table: {sum_table}")
        
    # prob_a = calculate_probability('', 'a', tables)
    # print(f"Probability of observing 'a': {prob_a}")
    # prob_b = calculate_probability('', 'b', tables)
    # print(f"Probability of observing 'b': {prob_b}")
    # prob_c = calculate_probability('', 'c', tables)
    # print(f"Probability of observing 'c': {prob_c}")
    
    # prob_aa = calculate_probability('a', 'a', tables)
    # print(f"Probability of observing 'aa': {prob_aa}")
    
    vocabulary = set(tables[0])
    current_sequence = initial_sequence

    for _ in range(k):
        # Predict the most likely next character
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")

    
if __name__ == "__main__":
    main()
