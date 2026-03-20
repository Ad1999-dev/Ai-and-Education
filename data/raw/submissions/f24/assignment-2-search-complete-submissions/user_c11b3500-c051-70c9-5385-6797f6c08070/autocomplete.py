from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.isLast = False
        self.freq = 1
        

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                    if char not in node.children:
                        node.children[char] = Node()
                    node.children[char].freq += 1 #added this line
                    node = node.children[char]
            node.isLast = True            

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        auto = []
        for char in prefix: #Find last input node in tree
            if char not in node.children:
                start = char
                return
            node = node.children[char]
        if node.isLast:
            auto.append('')
        FIFO = [(node, '')]
        while FIFO: 
            current = FIFO.pop(0)
            if current[0].isLast:
                auto.append(current[1])
            for char, node in current[0].children.items():  
                FIFO.append((node, current[1] + char))
        return [prefix + suffix for suffix in auto]

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        auto = []
        for char in prefix:
            if char not in node.children:
                break
            node = node.children[char]
        #At this point I should be at the goal node
        LIFO = [(node, '')]
        if node.isLast:
            auto.append('')
        while LIFO:
            current = LIFO.pop()
            if current[0].isLast:
                auto.append(current[1])
            for char, node in current[0].children.items():
                LIFO.append((node, current[1] + char))
        return [prefix + suffix for suffix in auto]


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                break
            node = node.children[char]
        priority = [(node, '', 0)]
        auto = []
        
        while priority:
            current = priority.pop()
            if current[0].isLast:
                auto.append(current[1])
            next = []
            for char, node in current[0].children.items():
                next.append((node, current[1] + char, node.freq))
            next.sort(key=lambda node: node[2])
            for element in next:
                priority.append(element)
        return [prefix + suffix for suffix in auto]
