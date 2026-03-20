from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.path_cost = 0
    
    def __lt__(self, other):
        return self.path_cost < other.path_cost

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    def _get_inverse(self, node):
        for child in node.children.values():
            child.path_cost = 1.0 / (child.path_cost)
            self._get_inverse(child)
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                node.children[char].path_cost += 1
                node = node.children[char]
            node.is_word = True
        self._get_inverse(self.root)

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        suggest = []
        node = self.root
        
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggest
        
        queue = deque([(node, prefix)])
        while queue:
            node, current_prefix = queue.popleft()
            if node.is_word:
                suggest.append(current_prefix)
            for char, child_node in node.children.items():
                queue.append((child_node, current_prefix + char))
        return suggest

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggest = []
        node = self.root
        
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggest
        
        stack = deque([(node, prefix)])
        while stack:
            node, current_prefix = stack.popleft()
            if node.is_word:
                suggest.append(current_prefix)
            for char, child_node in node.children.items():
                stack.appendleft((child_node, current_prefix + char))
        return suggest


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        suggest = []
        node = self.root
        
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggest
        
        heap = [(node, prefix)]
        while heap:
            node, current_prefix = heapq.heappop(heap)
            if node.is_word:
                suggest.append(current_prefix)
            for char, child_node in node.children.items():
                child_node.path_cost += node.path_cost
                heapq.heappush(heap,(child_node, current_prefix + char))
        return suggest
            
