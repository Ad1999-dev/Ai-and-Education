from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)
done = False
while not done:
    prefix = input("enter prefix. control c to exit: ")
    
    print("BFS: ", autocomplete_engine.suggest_bfs(prefix))
    print("DFS: ", autocomplete_engine.suggest_dfs(prefix))
    print("UCS: ", autocomplete_engine.suggest_ucs(prefix))

# create_gui(autocomplete_engine)