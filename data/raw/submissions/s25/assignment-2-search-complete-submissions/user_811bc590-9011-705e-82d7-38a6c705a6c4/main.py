from autocomplete import Autocomplete
from utilities import read_file, create_gui

autocomplete_engine = Autocomplete()

# doc = "there though that the their through thee thou thought thag "
# autocomplete_engine.build_tree(doc)
# print("suggestions:", autocomplete_engine.suggest('t'))


filename = 'genZ.txt'
read_file(filename, autocomplete_engine)
create_gui(autocomplete_engine)


# doc = "air ball cat car card carpet carry cap cape"

# # doc_for_weight = "crawl cat catch cap cape chat"
# autocomplete_engine.build_tree(doc)

# # print(autocomplete_engine.root.children['c'].children['h'].children['a'].children['t'].path_cost)

# print("suggestions:", autocomplete_engine.suggest('c'))
