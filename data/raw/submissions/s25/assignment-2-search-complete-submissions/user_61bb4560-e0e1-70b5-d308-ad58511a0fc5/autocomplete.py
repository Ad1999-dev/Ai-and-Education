from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        frequency_count = {}

        words = document.split()
        for word in words:
            for char in word:
                if char not in frequency_count:
                    frequency_count[char] = 0
                frequency_count[char] += 1


        for word in words:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_word = True

        for word in words:
            node = self.root
            for char in word:
                node = node.children[char]
                node.frequency = frequency_count[char]  

        return frequency_count 

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No suggestions if prefix not found
            node = node.children[char]

        suggestions = []
        queue = deque([(node, prefix)])  # Start with the node corresponding to the prefix
        
        while queue:
            current_node, current_prefix = queue.popleft()
            
            if current_node.is_word:
                suggestions.append(current_prefix)
            
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char))
        
        return suggestions


    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No suggestions if prefix not found
            node = node.children[char]

        suggestions = []

        def dfs(curr_node, current_prefix):
            if curr_node.is_word:
                suggestions.append(current_prefix)
            for char, child in curr_node.children.items():
                dfs(child, current_prefix + char)

        dfs(node, prefix)
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No suggestions if prefix not found
            node = node.children[char]

        suggestions = []
        priority_queue = []

        # Step 1: UCS function using frequency-based cost
        def ucs(curr_node, current_prefix, cost):
            if curr_node.is_word:
                heapq.heappush(priority_queue, (cost, current_prefix))  # Add word with its cost
            for char, child in curr_node.children.items():
                # Step 2: Adjust cost based on frequency of the character
                char_cost = 1 / (child.frequency + 1)  # Higher frequency = lower cost
                new_cost = cost + char_cost
                ucs(child, current_prefix + char, new_cost)

        # Start UCS from the current node for the given prefix
        ucs(node, prefix, 0)

        # Step 3: Return the suggestions sorted by cost
        return [suffix for _, suffix in sorted(priority_queue, key=lambda x: x[0])]

