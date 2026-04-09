#'''
from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)
create_gui(autocomplete_engine)

#'''

# Use the following to test implementations:

'''
from autocomplete import Autocomplete

# Create an instance of Autocomplete.
autocomplete_engine = Autocomplete()

# Step 1: Load words from a test file instead of the original one.
filename = 'test.txt'  # Use test.txt for the test here

# Step 2: Read the contents of the test file and build the trie.
with open(filename, 'r') as file:
    document = file.read()  # Read the whole content of test.txt

    print("\n\n\n")
    print("Contents of test.txt:", document.strip())  # Print the contents for comparison
    print("\n")
    #print("Contents of the tree:")

    autocomplete_engine.build_tree(document)  # Build the trie from the content

# Step 3: Print the trie structure to verify if it works correctly.

autocomplete_engine.print_trie()
print("\n")
autocomplete_engine.print_trie_structure()

print("\n")
print(autocomplete_engine.suggest_bfs("the"))
print(autocomplete_engine.suggest_dfs("the"))
print(autocomplete_engine.suggest_ucs("the"))
   
print("\n")
print(autocomplete_engine.suggest_bfs("t"))
print(autocomplete_engine.suggest_dfs("t"))
print(autocomplete_engine.suggest_ucs("t"))

print("\n")
print(autocomplete_engine.suggest_bfs("foo"))
print(autocomplete_engine.suggest_dfs("foo"))
print(autocomplete_engine.suggest_ucs("foo"))

print("\n")
# Optional: You can later uncomment this for the GUI if you need to use it.
# create_gui(autocomplete_engine)

#'''


