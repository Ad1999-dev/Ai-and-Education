from autocomplete import Autocomplete
from utilities import read_file, create_gui

# Initialize the autocomplete engine
autocomplete_engine = Autocomplete()

# Specify the filename
filename = 'genZ.txt'

# Read the file and build the trie
read_file(filename, autocomplete_engine)

# Call suggest_bfs on the autocomplete_engine instance with the prefix 'li'
dfs_suggestions = autocomplete_engine.suggest_dfs('th')
bfs_suggestions = autocomplete_engine.suggest_bfs('th')
ucs_suggestions = autocomplete_engine.suggest_ucs('th')

# Print the suggestions for the prefix 'th'
print(f"Autocomplete suggestions using dfs for prefix 'th':")
print(dfs_suggestions)
# Print the suggestions for the prefix 'th'
print(f"Autocomplete suggestions using bfs for prefix 'th':")
print(bfs_suggestions)

print(f"Autocomplete suggestions using ucs for prefix 'th':")
print(ucs_suggestions)
# Create the GUI for user interaction
create_gui(autocomplete_engine)