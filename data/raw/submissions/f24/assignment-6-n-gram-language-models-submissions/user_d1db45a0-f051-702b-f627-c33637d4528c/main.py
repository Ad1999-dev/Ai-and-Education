from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char
import time

war_and_peace_examples = ["the moon", "lemonade", "pierre", "russian", "for"]
alice_in_wonderland_examples = ["the trees", "assemble", "alice", "rabbit", "for"]

def main():
    document = read_file("warandpeace.txt")
    n = int(input("Enter the number of grams (n): "))
    #initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))

    start_time = time.time()
    tables = create_frequency_tables(document, n)
    print("table creation time: " + str(time.time() - start_time))

    # print("TABLES:")
    # print(tables)

    vocabulary = set(tables[0])

    for initial_sequence in war_and_peace_examples:
        print("testing with initial sequence: " + initial_sequence)
        current_sequence = initial_sequence
        for _ in range(k):
            # Predict the most likely next character
            next_char = predict_next_char(current_sequence, tables, vocabulary)
            current_sequence += next_char

        print(f"Updated sequence: {current_sequence}")
if __name__ == "__main__":
    main()
