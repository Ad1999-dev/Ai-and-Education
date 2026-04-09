from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        # self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                pass

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        pass

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        pass


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        pass
