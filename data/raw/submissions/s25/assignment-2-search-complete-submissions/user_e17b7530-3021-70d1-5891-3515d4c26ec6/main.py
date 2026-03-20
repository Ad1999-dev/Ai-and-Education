from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = 'test.txt'
read_file(filename, autocomplete_engine)
autocomplete_engine.print_tree()
create_gui(autocomplete_engine)