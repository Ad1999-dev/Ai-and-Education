from collections import deque
import heapq
import queue
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.cost = 1

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_word = True
        
        for word in document.split():
            node = self.root
            for char in word:
                node.cost += 1
                node = node.children[char]

        self.calcCost(self.root)

    def calcCost(self, node):
        if node.cost != 1.0:
            node.cost = 1.0 / node.cost
        for child in node.children.values():
            self.calcCost(child)

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        auto = []
        queue = deque([(node,prefix)])
        while queue:
            curnode, nextpref = queue.popleft()
            if curnode.is_word:
                auto.append(nextpref)
            for char, nextnode in curnode.children.items():
                queue.append((nextnode, nextpref + char))
        return auto


    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        auto = []
        stack = deque([(node,prefix)])
        while stack:
            curnode, nextpref = stack.pop()
            if curnode.is_word:
                auto.append(nextpref)
            for char, nextnode in curnode.children.items():
                stack.append((nextnode, nextpref + char))
        return auto


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        auto = []
        PQUEUE = [(node.cost, prefix, node)]
        heapq.heapify(PQUEUE)

        while PQUEUE:
            cost, nextpref, curnode = heapq.heappop(PQUEUE)

            if curnode.is_word:
                auto.append(nextpref)

            for char, nextnode in curnode.children.items():
                heapq.heappush(PQUEUE, (nextnode.cost, nextpref + char, nextnode))

        return auto
