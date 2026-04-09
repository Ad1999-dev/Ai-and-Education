from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = '/Users/<redacted>/Documents/CS383_Projects/assignment-2-search-complete-<redacted>/genZ.txt'
read_file(filename, autocomplete_engine)
create_gui(autocomplete_engine)