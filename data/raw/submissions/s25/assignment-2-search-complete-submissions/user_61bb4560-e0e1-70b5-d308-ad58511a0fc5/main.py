from autocomplete import Autocomplete
from utilities import read_file, create_gui

# Create an instance of the Autocomplete class
autocomplete_engine = Autocomplete()

# Specify the filename of the text document to read
filename = 'genZ.txt'

# Read the file and populate the autocomplete tree
read_file(filename, autocomplete_engine)

# You can now call suggest_bfs to test the suggestions for a specific prefix
prefix = "th"  # Define the prefix you want to test
suggestions = autocomplete_engine.suggest_ucs(prefix)  # Call the suggest_bfs method

# Print the suggestions obtained from the suggest_bfs method
print(f"Suggestions for prefix '{prefix}': {suggestions}")

# Create the GUI (this part remains unchanged)
create_gui(autocomplete_engine)
