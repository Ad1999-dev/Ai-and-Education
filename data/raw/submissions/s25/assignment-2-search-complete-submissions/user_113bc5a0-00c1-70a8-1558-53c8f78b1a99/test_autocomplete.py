from autocomplete import Autocomplete

def test_autocomplete_methods():
    # Create test instance
    ac = Autocomplete()
    
    # Test data
    test_words = "cat catch cats dog dogs done deal deer deep"
    ac.build_tree(test_words)
    
    # Test each method with different prefixes
    prefixes = ["c", "ca", "cat", "d", "do", "de", "x"]
    
    print("Testing BFS method:")
    ac.suggest = ac.suggest_bfs
    for prefix in prefixes:
        print(f"Prefix '{prefix}': {ac.suggest(prefix)}")
    
    print("\nTesting DFS method:")
    ac.suggest = ac.suggest_dfs
    for prefix in prefixes:
        print(f"Prefix '{prefix}': {ac.suggest(prefix)}")
    
    print("\nTesting UCS method:")
    ac.suggest = ac.suggest_ucs
    for prefix in prefixes:
        print(f"Prefix '{prefix}': {ac.suggest(prefix)}")

def test_specific_case():
    ac = Autocomplete()
    
    # Build a small, specific tree
    test_word = "hello help helps helper helping"
    ac.build_tree(test_word)
    
    # Test with "hel" prefix using all methods
    prefix = "hel"
    
    ac.suggest = ac.suggest_bfs
    print(f"\nBFS for 'hel': {ac.suggest(prefix)}")
    
    ac.suggest = ac.suggest_dfs
    print(f"DFS for 'hel': {ac.suggest(prefix)}")
    
    ac.suggest = ac.suggest_ucs
    print(f"UCS for 'hel': {ac.suggest(prefix)}")

if __name__ == "__main__":
    print("Running all tests...")
    test_autocomplete_methods()
    print("\nRunning specific test case...")
    test_specific_case()