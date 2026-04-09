from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)

print(autocomplete_engine.suggest_ucs("th"))

create_gui(autocomplete_engine)