from utilities import read_file, print_table
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char
import time
import sys

def test_corpus_and_n_values():
    """
    Test the N-gram model with different corpus files and n values.
    Compare and report performance metrics.
    """
    # List of corpus files to test
    corpus_files = [
        'Alice\'s Adventures in Wonderland.txt',
        'warandpeace.txt'
    ]
    
    # List of n values to test
    n_values = [1, 2, 3, 4, 5, 15, 30, 75]  # You can adjust these as needed
    
    # Test phrases to complete
    test_phrases = [
        "The",
        "She",
        "And then",
        "It was"
    ]
    
    # Number of characters to generate
    completion_length = 30
    
    print("\n===== N-GRAM MODEL TESTING =====\n")
    
    # Store results for comparison
    results = {}
    
    for corpus_file in corpus_files:
        print(f"\n----- Testing with corpus: {corpus_file} -----")
        
        # Read the document
        document = read_file(corpus_file)
        doc_size = len(document)
        print(f"Corpus size: {doc_size} characters")
        
        results[corpus_file] = {}
        
        for n in n_values:
            print(f"\n  Testing with n = {n}")
            
            # Measure table creation time
            start_time = time.time()
            try:
                tables = create_frequency_tables(document, n)
                table_time = time.time() - start_time
                
                # Get vocabulary
                vocabulary = set(tables[0].keys())
                print(f"  Vocabulary size: {len(vocabulary)} unique characters")
                
                # Memory estimate based on table size
                table_sizes = [sum(len(char_dict) for char_dict in table.values()) for table in tables]
                print(f"  Table creation time: {table_time:.2f} seconds")
                print(f"  Approximate entries in tables: {sum(table_sizes)}")
                
                # Test prediction performance
                completion_results = []
                
                for phrase in test_phrases:
                    start_time = time.time()
                    current = phrase
                    
                    for _ in range(completion_length):
                        next_char = predict_next_char(current, tables, vocabulary)
                        current += next_char
                    
                    completion_time = time.time() - start_time
                    print(f"\n  Starting with: '{phrase}'")
                    print(f"  Completed as: '{current}'")
                    print(f"  Completion time: {completion_time:.4f} seconds")
                    
                    completion_results.append({
                        'phrase': phrase,
                        'completion': current,
                        'time': completion_time
                    })
                
                # Store results
                results[corpus_file][n] = {
                    'table_time': table_time,
                    'table_size': sum(table_sizes),
                    'completions': completion_results
                }
                
            except MemoryError:
                print(f"  Memory error occurred with n = {n} - too large for available memory")
                results[corpus_file][n] = 'Memory Error'
            except Exception as e:
                print(f"  Error occurred: {str(e)}")
                results[corpus_file][n] = f'Error: {str(e)}'
    
    # Print summary of results
    print("\n\n===== TESTING SUMMARY =====")
    
    for corpus_file in corpus_files:
        print(f"\n----- Summary for {corpus_file} -----")
        for n in n_values:
            if isinstance(results[corpus_file].get(n), dict):
                res = results[corpus_file][n]
                print(f"n = {n}:")
                print(f"  Table creation time: {res['table_time']:.2f} seconds")
                print(f"  Table size: {res['table_size']} entries")
                avg_time = sum(c['time'] for c in res['completions']) / len(res['completions'])
                print(f"  Average completion time: {avg_time:.4f} seconds")
            else:
                print(f"n = {n}: {results[corpus_file].get(n, 'Not tested')}")

def main():
    # Check if we're in testing mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_corpus_and_n_values()
        return
    
    # Original interactive mode
    document = read_file('warandpeace.txt')
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)
    vocabulary = set(tables[0])
    
    current_sequence = initial_sequence

    for _ in range(k):
        # Predict the most likely next character
        next_char = predict_next_char(current_sequence, tables, vocabulary)
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")

if __name__ == "__main__":
    main()