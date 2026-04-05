from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char
import functools
import operator 

def main():
    document = read_file('alice.txt')
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
    # f1 = open('test.txt','r')
    # freqtable = {}
    # print(f1.readline()[0:3])
    # print(create_frequency_tables('test.txt', 3))
    # x = functools.reduce(operator.add, create_frequency_tables('test.txt', 3)[0].values())
    # print(x)
    
    tex = read_file('test2.txt')
    print(len(tex))
    r = calculate_probability('aa', 'c', create_frequency_tables(tex, 3))
    print(r)
    t = (create_frequency_tables(tex, 3))
    print(t)
    print(predict_next_char('aa', t, {'a', 'b', 'c'}))