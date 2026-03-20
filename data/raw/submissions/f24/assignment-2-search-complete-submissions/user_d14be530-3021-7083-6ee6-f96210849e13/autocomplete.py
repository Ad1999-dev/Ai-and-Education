from collections import deque, defaultdict, OrderedDict
import heapq
import random
import string
import math

class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.char_frequency = defaultdict(int)  # Frequency count for each character
        self.frequency = 0 # Total frequency of this node

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs  #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            word = word.lower()
            node = self.root
            node.frequency += 1  # Increment frequency at root
            for char in word:
                node.char_frequency[char] += 1
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.frequency += 1  # Increment frequency at each node
            node.is_end_of_word = True



    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!

    def _find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node



    def suggest_bfs(self, prefix):
        node = self._find_node(prefix)
        if not node:
            return []  # If prefix is not found, return an empty list

        suggestions = []
        queue = deque([(node, prefix)])

        while queue:
            current_node, current_word = queue.popleft()

            # Check if the current node marks the end of a word
            if current_node.is_end_of_word:
                suggestions.append(current_word)

            # Add all children to the queue, appending the character to the current word
            for char in sorted(current_node.children.keys()):
                child_node = current_node.children[char]
                queue.append((child_node, current_word + char))

        return suggestions
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self._find_node(prefix)
        if not node:
            return []

        suggestions = []
        self._dfs(node, prefix, suggestions)
        return suggestions

    def _dfs(self, node, current_word, suggestions):
        if node.is_end_of_word:
            suggestions.append(current_word)

        # Traverse children in insertion order
        for char, child_node in node.children.items():
            self._dfs(child_node, current_word + char, suggestions)





    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self._find_node(prefix)
        if not node:
            return []

        suggestions = []
        queue = [(0, prefix, node)]  # (path_cost, current_word, node)
        visited = {}

        while queue:
            cost, current_word, current_node = heapq.heappop(queue)

            # Skip if we've already visited with a lower cost
            if current_node in visited and visited[current_node] <= cost:
                continue
            visited[current_node] = cost

            if current_node.is_end_of_word:
                suggestions.append(current_word)

            for char, child_node in current_node.children.items():
                frequency = current_node.char_frequency[char]
                if frequency == 0:
                    continue

                # Calculate path cost using inverse frequency
                path_cost = cost + (1 / frequency)
                # Push onto heap with updated tuple structure
                heapq.heappush(queue, (path_cost, current_word + char, child_node))

        return suggestions
