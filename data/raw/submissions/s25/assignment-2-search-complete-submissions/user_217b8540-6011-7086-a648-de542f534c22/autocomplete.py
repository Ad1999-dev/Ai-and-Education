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
        self.frequency = 0  

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()  
                node = node.children[char]
                node.frequency += 1  
            node.is_word = True  

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        if not prefix:  # Return empty list if prefix is empty
            return []

        node = self._get_prefix_node(prefix)
        if not node:
            return []

        suggestions = []
        queue = deque([(node, prefix)])  # Queue of (node, current_word)

        while queue:
            current_node, current_word = queue.popleft()
            if current_node.is_word:
                suggestions.append(current_word)

            for char, child_node in current_node.children.items():
                queue.append((child_node, current_word + char))

        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):

        if not prefix:  # Return empty list if prefix is empty
            return []

        node = self._get_prefix_node(prefix)
        if not node:
            return []

        suggestions = []
        stack = [(node, prefix)]  # Stack of (node, current_word)

        while stack:
            current_node, current_word = stack.pop()
            if current_node.is_word:
                suggestions.append(current_word)

            # Push children in reverse order to maintain insertion order
            for char, child_node in reversed(list(current_node.children.items())):
                stack.append((child_node, current_word + char))

        return suggestions



    #TODO for students!!!
    def suggest_ucs(self, prefix):
        if not prefix:  # Return empty list if prefix is empty
            return []

        node = self._get_prefix_node(prefix)
        if not node:
            return []

        suggestions = []
        heap = []  # Priority queue of (cost, tiebreaker, node, current_word)
        counter = itertools.count()  # Unique counter for tiebreaking

        # Push the initial node onto the heap
        heapq.heappush(heap, (0, next(counter), node, prefix))

        while heap:
            cost, _, current_node, current_word = heapq.heappop(heap)
            if current_node.is_word:
                suggestions.append(current_word)

            for char, child_node in current_node.children.items():
                # Cost is the inverse of frequency (lower cost for higher frequency)
                new_cost = cost + (1 / child_node.frequency)
                heapq.heappush(heap, (new_cost, next(counter), child_node, current_word + char))

        return suggestions
        

    def _get_prefix_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
