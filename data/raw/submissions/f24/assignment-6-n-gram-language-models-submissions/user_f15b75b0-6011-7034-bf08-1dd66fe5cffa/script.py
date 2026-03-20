from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    document = "aababcaccaaacbaabcaa"
    n = 3

    tables = create_frequency_tables(document, n)

    vocabulary = set(document)

    sequence = "aa"

    print("Frequency Tables:")
    print_table(tables, n - 1)

    print("\nProbabilities:")
    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        print(f"P({char} | {sequence}) = {prob:.4f}")

    predicted_char = predict_next_char(sequence, tables, vocabulary)
    print(f"\nPredicted next character: {predicted_char}")

if __name__ == "__main__":
    main()
