from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = 'ml.txt'
read_file(filename, autocomplete_engine)
create_gui(autocomplete_engine)