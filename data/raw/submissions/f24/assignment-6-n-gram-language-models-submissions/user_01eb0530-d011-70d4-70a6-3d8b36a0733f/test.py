from collections import defaultdict
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def main():
    # Training document and parameters
    document = "aababcaccaaacbaabcaa"
    n = 3  # for trigrams
    vocabulary = set(document)  # Get vocabulary from document
    
    # Create frequency tables
    tables = create_frequency_tables(document, n)
    
    # Print raw frequency tables
    print("\nRaw Frequency Tables:")
    for i, table in enumerate(tables):
        print(f"\nTable {i+1} ({i+1}-grams):")
        for sequence, count in sorted(table.items()):
            print(f"f({sequence}) = {count}")
    
    # Test sequence prediction
    test_sequence = "aa"  # Make sure this is a string
    print(f"\nProbabilities for next character after '{test_sequence}':")
    
    # Calculate probabilities for each possible next character
    for char in sorted(vocabulary):
        prob = calculate_probability(test_sequence, char, tables)  # Fixed line
        print(f"P({char}|{test_sequence}) = {prob}")
    
    # Make prediction
    predicted = predict_next_char(test_sequence, tables, vocabulary)
    print(f"\nPredicted next character after '{test_sequence}': '{predicted}'")

if __name__ == "__main__":
    main()
