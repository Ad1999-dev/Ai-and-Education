from collections import deque
from collections import defaultdict
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.path_cost = float('inf')
        self.frequency = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        # self.char_frequency = defaultdict(int)
        if document:
            self.build_tree(document)
    
    
    def build_tree(self, document):
        # Count frequencies of characters following prefixes
        for word in document.lower().split():
            curr = self.root
            # cumulative_cost = 0
            for char in word:
                if char not in curr.children:
                    curr.children[char] = Node()
                    # Set the path cost as the inverse of the character frequency
                else:
                    curr.children[char].frequency += 1
                    # frequency = self.char_frequency[char]
                curr = curr.children[char]
            curr.is_end = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def find_node(self, prefix):
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def suggest_bfs(self, prefix):
        start_node = self.find_node(prefix.lower())
        if not start_node:
            return []  # Return empty list if prefix not found
        
        suggestions = []
        queue = deque([(start_node, prefix.lower())])
        
        while queue:
            current_node, current_prefix = queue.popleft()
            if current_node.is_end:
                suggestions.append(current_prefix)
            
            # Enqueue all children nodes
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char))
        
        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        start_node = self.find_node(prefix.lower())
        suggestions = []
        if not start_node:
            return suggestions  # Return empty list if prefix not found
        
        def dfs(node, current_prefix):
            if node.is_end:
                suggestions.append(current_prefix)
            
            for char, child_node in node.children.items():
                dfs(child_node, current_prefix + char)
        
        dfs(start_node, prefix.lower())
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        start_node = self.find_node(prefix)
        search = self.root
        if not start_node:
             return []
        for char in prefix:
            if char not in search.children:
                return []
            search = search.children[char]
        heapUCS = []
        heapq.heappush(heapUCS, (0, prefix, search))
        priorityQueue = []

        while heapUCS:
            cost, word, node = heapq.heappop(heapUCS)
            if node.is_end:
                priorityQueue.append(word)
            for char, child in node.children.items():
                heapq.heappush(heapUCS, (cost - child.frequency, word + char, child))
        return priorityQueue
        
# for word in document.split():
#             node = self.root
#             for char in word:
#                 if char not in node.children:
#                   node.children[char] = Node()
#                 node = node.children[char]
#             node.is_end = True  # Mark the end of the word
