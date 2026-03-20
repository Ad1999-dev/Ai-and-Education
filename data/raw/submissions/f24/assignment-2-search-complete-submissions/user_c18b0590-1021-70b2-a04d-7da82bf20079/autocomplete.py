from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self, letter = ""):
        self.letter = letter
        self.end = False
        self.children = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node(char)
                node = node.children[char]
            node.end = True
                
        
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):

        suggestions = []
        node = self.root

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions  

        def dfs(current_node, current_prefix):
            if current_node.end:
                suggestions.append(current_prefix)

            for child in current_node.children.values():
                dfs(child, current_prefix + child.letter)

        dfs(node, prefix)
        return suggestions
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggestions = []
        node = self.root

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions 

        def dfs(current_node, current_prefix):
            if current_node.end:
                suggestions.append(current_prefix)

            for child in current_node.children.values():
                dfs(child, current_prefix + child.letter)

        dfs(node, prefix)
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        suggestions = []
        node = self.root

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions  

        heap = []
        queue = []
        node_queue = [(node, prefix)]

        while node_queue:
            current_node, current_prefix = node_queue.pop(0)

            if current_node.end:
                heapq.heappush(heap, current_prefix)

            for child in current_node.children.values():
                node_queue.append((child, current_prefix + child.letter))

        while heap:
            suggestions.append(heapq.heappop(heap))

        return suggestions

a = Autocomplete("genZ.txt")
a.build_tree("genZ.txt")
print(a.root.children)
