from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)
#create_gui(autocomplete_engine)

test_prefix = "th" 
suggestions = autocomplete_engine.suggest_bfs(test_prefix)
print(f"Suggestions for '{test_prefix}': {suggestions}")
 
suggestions = autocomplete_engine.suggest_dfs(test_prefix)
print(f"Suggestions for '{test_prefix}': {suggestions}")

suggestions = autocomplete_engine.suggest_ucs(test_prefix)
print(f"Suggestions for '{test_prefix}': {suggestions}")