from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.isword = False
        

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs
    
    def build_tree(self, document): 
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children: #if the chararcter isn't already a child of a node but it's in a word then a new node is made
                    node.children[char] = Node()
                node = node.children[char]
            node.isword = True #once all the characters of a word are made into nodes then that node is the end of word

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        #traverse the tree until it's at the end of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  #If there is no prefix, then return an empty list
        
        suggestions = []  
        queue = deque([(node, prefix)]) #queue stores (current_node, current_word)
        
        while queue:
            current_node, current_word = queue.popleft()
            
            if current_node.isword:
                suggestions.append(current_word)

            for char, child in current_node.children.items():
                queue.append((child, current_word + char))
                
        return suggestions


    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        #traverse the tree until it's at the end of the prefix
        for char in prefix:    
            if char in node.children:
                node = node.children[char]
            else:
                return []  #If there is no prefix, then return an empty list
        
        suggestions = []  
        stack = [(node, prefix)]  #stack stores (current_node, current_word)
        
        while stack:
            current_node, current_word = stack.pop()
            
            if current_node.isword:
                suggestions.append(current_word)
    
            for char in reversed(current_node.children.keys()):
                stack.append((current_node.children[char], current_word + char))

        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:    
            if char in node.children:
                node = node.children[char]
            else:
                return []  #If there is no prefix, then return an empty list
        
        suggestions = []  # List to store words in UCS order
        priority_queue = []  # heap to store (cumulative cost, current_word, current_node)
        heapq.heappush(priority_queue, (0, prefix, node))
        
        visited = [] # list to keep track of visited nodes

        while priority_queue:
            cumulative_cost, current_word, current_node = heapq.heappop(priority_queue)
            
            if current_node.isword:
                suggestions.append(current_word)  # Add word
                #suggestions.append((current_word, cumulative_cost))

            # Enqueuing the child nodes
            for char, child in current_node.children.items():
                if child not in visited:
                    cost = 1 / len(current_word + char) 
                    heapq.heappush(priority_queue, (cumulative_cost + cost, current_word + char, child))
                    visited.append(child)
        
        return suggestions
