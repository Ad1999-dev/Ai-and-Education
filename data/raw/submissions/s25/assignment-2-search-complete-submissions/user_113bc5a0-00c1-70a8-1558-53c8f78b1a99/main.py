from autocomplete import Autocomplete
from utilities import read_file, create_gui

# Create an instance of Autocomplete
autocomplete_engine = Autocomplete()

# Read the file and build the trie
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)

# Create the GUI for user interaction
create_gui(autocomplete_engine)

# This part can be removed or commented out
# because you're already using read_file.
# If you wish to display suggestions while building the GUI, you can handle that through the GUI itself.
# However, if you want to print suggestions for "th" immediately upon launching, do so before creating the GUI:
suggestions = autocomplete_engine.suggest_ucs("th")
print("Autocomplete suggestions for 'th':")
for suggestion in suggestions:
    print(suggestion)

