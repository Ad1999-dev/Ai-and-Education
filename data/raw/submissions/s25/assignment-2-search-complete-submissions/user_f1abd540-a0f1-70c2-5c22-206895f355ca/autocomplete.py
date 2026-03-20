from collections import defaultdict, deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.edge_cost=1

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
                    node.children[char]= Node()
                node = node.children[char]
            node.is_word=True

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

        l = []
        queue = deque([(node, prefix)]) 
        
        while queue:
            current_node, current_prefix = queue.popleft()
            if current_node.is_word:
                l.append(current_prefix)
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char))
        
        return l

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return [] 
            node = node.children[char]
        suggestions = []
        
        def dfs(current_node, word_so_far):
            if current_node.is_end:
                suggestions.append(word_so_far)
            for char, child in current_node.children.items():
                dfs(child, word_so_far + char)
        
        dfs(node, prefix)
        return suggestions
    

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]

        suggestions = []
        # Priority queue elements: (accumulated_cost, word_so_far, current_node)
        frontier = []
        heapq.heappush(frontier, (0, prefix, node))
        
        while frontier:
            cost, word_so_far, current_node = heapq.heappop(frontier)
            # Use is_word instead of is_end
            if current_node.is_word:
                suggestions.append(word_so_far)
            for char, child in current_node.children.items():
                # Assume each child node has an edge_cost attribute (set in build_tree)
                new_cost = cost + child.edge_cost  
                heapq.heappush(frontier, (new_cost, word_so_far + char, child))
        
        return suggestions