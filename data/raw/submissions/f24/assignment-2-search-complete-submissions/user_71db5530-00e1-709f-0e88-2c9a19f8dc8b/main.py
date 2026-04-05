from autocomplete import Autocomplete
from utilities import read_file, create_gui
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', default= '',type=str, help='Algorithm to be used')
args = parser.parse_args()
autocomplete_engine = Autocomplete(algo=args.a)
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)
create_gui(autocomplete_engine)