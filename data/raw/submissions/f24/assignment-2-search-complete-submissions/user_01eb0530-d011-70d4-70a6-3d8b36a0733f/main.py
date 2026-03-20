from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()
filename = 'genZ.txt'
read_file(filename, autocomplete_engine)

prefix = "th"

bfs_suggestions = autocomplete_engine.suggest_bfs(prefix)
print("BFS:", bfs_suggestions)

dfs_suggestions = autocomplete_engine.suggest_dfs(prefix)
print("DFS:", dfs_suggestions)

ucs_suggestions = autocomplete_engine.suggest_ucs(prefix)
print("UCS:", ucs_suggestions)

create_gui(autocomplete_engine)
