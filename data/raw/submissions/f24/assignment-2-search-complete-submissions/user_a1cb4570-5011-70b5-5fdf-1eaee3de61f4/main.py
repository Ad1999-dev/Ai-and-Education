from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)
create_gui(autocomplete_engine)
prefix = "s"
suggestions = autocomplete_engine.suggest(prefix)
print(f"Suggestions for prefix '{prefix}':")
for suggestion in suggestions:
    print(suggestion)