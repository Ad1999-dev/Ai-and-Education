from collections import deque
import heapq
import random
import string

class Node:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.cost = 0  # Used for UCS to store inverse frequency

class Autocomplete:
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs  # Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        freq_map = {}  # To track frequencies of each node
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                freq_map[node] = freq_map.get(node, 0) + 1
            node.is_word = True

        # Store inverse frequencies as costs for UCS
        for node, freq in freq_map.items():
            node.cost = 1 / freq

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        # Navigate to the node corresponding to the last character of the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No suggestions available
            node = node.children[char]

        # Perform BFS from this node
        queue = deque([(node, prefix)])
        suggestions = []
        while queue:
            current_node, path = queue.popleft()
            if current_node.is_word:
                suggestions.append(path)
            for char, child in current_node.children.items():
                queue.append((child, path + char))
        return suggestions

    def suggest_dfs(self, prefix):
        # Navigate to the node corresponding to the last character of the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No suggestions available
            node = node.children[char]

        # Perform DFS from this node
        stack = [(node, prefix)]
        suggestions = []
        while stack:
            current_node, path = stack.pop()
            if current_node.is_word:
                suggestions.append(path)
            for char, child in current_node.children.items():
                stack.append((child, path + char))
        return suggestions

    def suggest_ucs(self, prefix):
        # Navigate to the node corresponding to the last character of the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No suggestions available
            node = node.children[char]

        # Use a priority queue for UCS
        heap = []
        counter = 0  # Unique identifier to break ties
        heapq.heappush(heap, (node.cost, counter, node, prefix))
        counter += 1
        suggestions = []
        while heap:
            cost, _, current_node, path = heapq.heappop(heap)
            if current_node.is_word:
                suggestions.append(path)
            for char, child in current_node.children.items():
                heapq.heappush(heap, (cost + child.cost, counter, child, path + char))
                counter += 1
        return suggestions
