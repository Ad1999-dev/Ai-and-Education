
from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = r"C:\Users\<redacted>\Desktop\Assingment 2_1\assignment-2-search-complete-<redacted>\test.txt"
read_file(filename, autocomplete_engine)
create_gui(autocomplete_engine)