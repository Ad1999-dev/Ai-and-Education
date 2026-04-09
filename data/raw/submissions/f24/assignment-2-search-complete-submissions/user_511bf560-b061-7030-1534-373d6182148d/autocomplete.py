from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.path_cost = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        self.char_frequencies = {}

        if document:
            self.build_tree(document)

    def calc_char_frequencies(self, document):
        self.char_frequencies = {}
        words = document.split()
        for word in words:
            for char in word:
                if char in self.char_frequencies:
                    self.char_frequencies[char] +=1
                else:
                    self.char_frequencies[char] = 1
            
    def set_path_cost(self, node, char, freq):
        if freq > 0:
            node.path_cost = 1/freq

    def build_tree(self, document):
        self.calc_char_frequencies(document) 
        for word in document.split():
            node = self.root
            for char in word.lower():
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_end_of_word =True #marks end of a word

        #setting path costs based on inverse frequencies 
        for char, freq in self.char_frequencies.items():
            if char in self.root.children:
                self.set_path_cost(self.root.children[char], char, freq)

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        node = self.root
        prefix = prefix.lower()
        #traversing to find the last node in the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return [] #this would mean prefix not found
        
        suggestions = []
        #bfs implementation with queue
        queue = deque([(node, prefix)])
        while queue:
            current_node, current_prefix = queue.popleft()
            if current_node.is_end_of_word:
                suggestions.append(current_prefix)

            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char))
        
        return suggestions

    
    def suggest_dfs(self, prefix):
        node = self.root
        prefix = prefix.lower()
        #traversing to find the last node in prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return [] #this would mean prefix not found

        suggestions =[]
        #dfs implementation with stack
        stack = [(node, prefix)]
        while stack:
            current_node, current_prefix = stack.pop()
            if current_node.is_end_of_word:
                suggestions.append(current_prefix)
                
            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))
        
        return suggestions


    def suggest_ucs(self, prefix):
        node = self.root
        prefix = prefix.lower()
        #traversing to find the last node in prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  #this would mean prefix not found

        suggestions = []
        #ucs implementation with priority queue heapq
        priority_queue = []  
        heapq.heappush(priority_queue, (0, prefix, node))
        while priority_queue:
            current_cost, current_prefix, current_node = heapq.heappop(priority_queue)
            if current_node.is_end_of_word:
                suggestions.append(current_prefix)
            for char, child_node in current_node.children.items():
                heapq.heappush(priority_queue, (current_cost + child_node.path_cost, current_prefix + char, child_node))

        return suggestions
