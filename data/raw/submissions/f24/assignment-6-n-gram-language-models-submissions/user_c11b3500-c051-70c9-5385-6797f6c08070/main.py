from utilities import read_file, print_table
import unittest
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char
"""
def main():
    document = read_file('warandpeace.txt')
    n = int(input("Enter the number of grams (n): "))
    initial_sequence = input(f"Enter an initial sequence: ")
    k = int(input("Enter the length of completion (k): "))
    
    tables = create_frequency_tables(document, n)
    #print(tables[1])

    vocabulary = set(tables[0])
    
    current_sequence = initial_sequence

    for _ in range(k):
        #Predict the most likely next character
        next_char = predict_next_char(current_sequence[-n:], tables, vocabulary)
        current_sequence += next_char      
        print(f"Updated sequence: {current_sequence}")
    """
class TestCalculateProbability(unittest.TestCase):

    def setUp(self):
        # Create a simple mock of the n-gram frequency tables
        self.tables = [
            {'a': 8, 'b': 4, 'c': 4, 'd': 2},      # 1-gram frequencies
            {'ab': 3, 'bc': 1, 'cd': 1, 'dc': 1, 'cb': 1, 'ba': 3, 'aa': 3, 'cc': 1},  # 2-gram frequencies
            {'abc': 1, 'bcd': 1, 'dcb': 1, 'cba': 1, 'aaa': 2, 'aba': 1, 'bab': 1},        # 3-gram frequencies
            {'abcd': 1, 'dcba': 1, 'aaaa': 1, 'abab': 1}                  # 4-gram frequencies
        ]

    def test_probability_valid(self):
        self.assertAlmostEqual(calculate_probability("zzz222222", "b", self.tables), None)
    """
    def test_probability_not_found_ngram(self):
        # Test input sequence not found in the n-gram table
        probability = calculate_probability("xyz", "l", self.tables)
        self.assertIsNone(probability)  # Should return None

    def test_probability_sequence_empty(self):
        # Test with an empty sequence
        probability = calculate_probability("", "l", self.tables)
        self.assertIsNone(probability)  # Should return None

    def test_probability_with_no_next_char(self):
        # Test when next_char is not frequent
        probability = calculate_probability("lo", "z", self.tables)  # "lo" is an existing 2-gram
        self.assertEqual(probability, 0.0)  # Should return 0 since 'loz' doesn't exist
    
    def test_probability_case_1(self):
        # Test valid input (sequence "ab", next_char "c") with modified tables
        probability = calculate_probability("he", "l", self.tables)
        self.assertAlmostEqual(probability, 0.5)  # P('l' | 'he') = freq('hel') / freq('he') = 1 / 2 = 0.5

    def test_probability_case_2(self):
        # Test valid input (sequence "h", next_char "e")
        probability = calculate_probability("h", "e", self.tables)
        self.assertAlmostEqual(probability, 1.0)  # P('e' | 'h') = freq('he') / freq('h') = 1 / 1 = 1.0

    def test_probability_case_3(self):
        # Test valid input (sequence "e", next_char "l")
        probability = calculate_probability("e", "l", self.tables)
        self.assertAlmostEqual(probability, 1.0)  # P('l' | 'e') = freq('el') / freq('e') = 2 / 2 = 1.0
    """

if __name__ == "__main__":
    unittest.main()
