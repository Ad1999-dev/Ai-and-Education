from collections import deque
import heapq
import random
import string


class Node:
    #TODO possibly include more things to keep track of based on how i implement the suggest methods
    def __init__(self):
        self.children = {}
        self.end_of_word = False
        self.frequency = 0 


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:    
                    node.children[char] = Node()
                node = node.children[char]
                node.frequency += 1
            node.end_of_word =  True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else: 
                return []

        suggestions = []
        queue = deque([(node, prefix)])

        while queue:
            current_node, current_prefix = queue.popleft()

            if current_node.end_of_word: #it's a leaf node
                suggestions.append(current_prefix) #because current_prefix is a word now

            for char, child_node in current_node.children.items(): #else, for each child letter, append the new possible prefixes into queue
                queue.append((child_node, current_prefix + char))

            
        return suggestions

        
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else: 
                return []
        
        suggestions = []
        self._dfs(node, prefix, suggestions)
        return suggestions
    
    def _dfs(self, node, current_prefix, suggestions):
        if node.end_of_word:
            suggestions.append(current_prefix)
        
        for char, child_node in node.children.items():
            self._dfs(child_node, current_prefix + char, suggestions)
       
  


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return [] # If the prefix is not found
            
        suggestions = []
        priority_queue = []

        heapq.heappush(priority_queue, (0, prefix, node))

        while priority_queue:
            path_cost,  current_prefix, current_node = heapq.heappop(priority_queue)

            if current_node.end_of_word:
                suggestions.append(current_prefix)

            for char, child_node in current_node.children.items():
                if child_node.frequency > 0:
                    cost = 1.0 / child_node.frequency 
                else:
                    cost = float('inf') 
            
                new_prefix = current_prefix + char
                new_cost = path_cost + cost 

                heapq.heappush(priority_queue, (new_cost, new_prefix, child_node))
        
        return suggestions
