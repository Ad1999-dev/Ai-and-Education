from collections import deque
import heapq
import math
import random
import string


class Node:
    #TODO
    def __init__(self, parent = None, children = {}, prefix = "", pathCost = 1):
        self.parent = parent
        self.children = children
        self.prefix = prefix
        self.pathCost = pathCost
        self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char in node.children:
                    node = node.children[char]
                    node.pathCost = 1/(round(1/node.pathCost) + 1)
                else:
                    node.children[char]= Node(node, {}, node.prefix + char)
                    node = node.children[char]
            node.is_word = True
                    
                
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
    
    def search(self, prefix, my_deque, pop_func, use_sort):
        listOfWords = list()
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return list()
        my_deque.extend(node.children.items())
        if(use_sort): 
            temp_deque = deque(sorted(my_deque, key = lambda item:item[1].pathCost))
            my_deque.clear()
            my_deque += temp_deque
        while my_deque:
            nodeTuple = pop_func()
            if  nodeTuple[1].is_word == True:
                listOfWords.append(nodeTuple[1].prefix)
                nodeTuple[1].is_word == False
            for child_char, child_node in nodeTuple[1].children.items():
                my_deque.append((child_char, child_node))
                if(use_sort): 
                    temp_deque = deque(sorted(my_deque, key = lambda item:item[1].pathCost))
                    my_deque.clear()
                    my_deque += temp_deque
        return listOfWords
            
    #TODO for students!!!
    def suggest_bfs(self, prefix):
        my_deque = deque()
        return self.search(prefix, my_deque, my_deque.popleft, 0)
        

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        my_deque = deque()
        return self.search(prefix, my_deque, my_deque.pop, 0)

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        my_deque = deque()
        return self.search(prefix, my_deque, my_deque.popleft, 1)
