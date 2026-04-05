from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)
#create_gui(autocomplete_engine)

autocomplete_engine.build_tree(filename)
autocomplete_engine.suggest_dfs('be')