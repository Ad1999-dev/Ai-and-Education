from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()

        self.char_frequencies = {}  
        
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        if document:
            self.build_tree(document)

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                
                if char in self.char_frequencies:
                    self.char_frequencies[char] += 1
                else:
                    self.char_frequencies[char] = 1  
            
            node.children['*'] = None  
        


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        current_node = self.root
    
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return [] 

        suggestions = []
        queue = [(current_node, prefix)]  

        while queue:
            node, current_prefix = queue.pop(0)  

            if '*' in node.children: 
                suggestions.append(current_prefix)

            for char, child_node in node.children.items():
                if char != '*':  
                    queue.append((child_node, current_prefix + char))

        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        current_node = self.root
    
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return [] 

        suggestions = []
        queue = [(current_node, prefix)]  

        while queue:
            node, current_prefix = queue.pop() 

            if '*' in node.children: 
                suggestions.append(current_prefix)

            for char, child_node in node.children.items():
                if char != '*':  
                    queue.append((child_node, current_prefix + char))

        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        current_node = self.root
    
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []  

        suggestions = []
        queue = [(0, current_node, prefix)]  
        
        while queue:
            current_cost, node, current_prefix = heapq.heappop(queue)
            
            if '*' in node.children: 
                suggestions.append(current_prefix)

            for char, child_node in node.children.items():
                if child_node is not None: 
                    new_cost = current_cost + (1 / self.char_frequencies.get(char, 1))
                    heapq.heappush(queue, (new_cost, child_node, current_prefix + char))

        return suggestions

