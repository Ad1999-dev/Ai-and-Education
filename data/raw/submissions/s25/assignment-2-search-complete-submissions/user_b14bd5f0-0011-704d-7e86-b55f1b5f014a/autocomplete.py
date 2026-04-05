from collections import deque
import heapq
import random
import string
import itertools

class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.cost = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs

    def calculate_inverse_frequency(self, document):
        freq = {}
        for char in document:
            if char in freq:
                freq[char] += 1
            else:
                freq[char] = 1
        inverse_freq = {}
        for char, frequency in freq.items():
            inverse_freq[char] = 1 / frequency
        return inverse_freq
    
    
    def build_tree(self, document):
        inverse_freq = self.calculate_inverse_frequency(document)

        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                node.children[char] = node.children.get(char, Node()) 
                node = node.children[char]
                node.cost = inverse_freq[char]
                
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        # Start from the root node and traverse to the end of the prefix
        if(prefix == ""): return []
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        # once last charachter in prefix is reached, start bfs
        queue = deque([(node, prefix)])
        suggestions = []
        while queue:
            current_node, current_prefix = queue.popleft()
            if current_node.is_word:
                suggestions.append(current_prefix)
            for char, child in current_node.children.items():
                queue.append((child, current_prefix + char))
        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        # Start from the root node and traverse to the end of the prefix
        if(prefix == ""): return []
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        # once last charachter in prefix is reached, start dfs
        stack = [(node, prefix)]
        suggestions = []
        while stack:
            current_node, current_prefix = stack.pop()
            if current_node.is_word:
                suggestions.append(current_prefix)
            for char, child in current_node.children.items():
                stack.append((child, current_prefix + char))
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        # Traverse to the node corresponding to the prefix
        if(prefix == ""): return []
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        priority_queue = [(0, prefix, node)]
        suggestions = []
        
        while priority_queue:
            current_cost, current_prefix, current_node = heapq.heappop(priority_queue)
            
            # If the current node represents a complete word, add it to suggestions
            if current_node.is_word:
                suggestions.append(current_prefix)
            
            # Explore children
            for char, child in current_node.children.items():
                new_cost = current_cost + child.cost
                new_prefix = current_prefix + char
                heapq.heappush(priority_queue, (new_cost, new_prefix, child))
        
        return suggestions