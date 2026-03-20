from autocomplete import Autocomplete
from utilities import read_file, create_gui

x = (5, "s", 3)
print(x[0])
autocomplete_engine = Autocomplete()
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)
create_gui(autocomplete_engine)