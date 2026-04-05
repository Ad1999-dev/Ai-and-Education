from collections import deque
import heapq
import random
import string
import itertools


class Node:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.freq = 0   # frequency count for the character (edge from parent)

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        if document:
            self.build_tree(document)
        self.suggest = self.suggest_random  # Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.freq += 1
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def _find_prefix_node(self, prefix):
        # Helper method to locate the node corresponding to the end of the prefix.
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node

    def suggest_bfs(self, prefix):
        start = self._find_prefix_node(prefix)
        if not start:
            return []
        suggestions = []
        queue = deque([(start, prefix)])
        while queue:
            node, word = queue.popleft()
            if node.is_word:
                suggestions.append(word)
            for char, child in node.children.items():
                queue.append((child, word + char))
        return suggestions

    def suggest_dfs(self, prefix):
        start = self._find_prefix_node(prefix)
        if not start:
            return []
        suggestions = []
        def dfs(node, word):
            if node.is_word:
                suggestions.append(word)
            for char, child in node.children.items():
                dfs(child, word+char)
        dfs(start, prefix)
        return suggestions

    def suggest_ucs(self, prefix):
        start = self._find_prefix_node(prefix)
        if not start:
            return []
        suggestions = []
        counter = itertools.count()  # unique counter for tie-breaking
        heap = [(0, next(counter), start, prefix)]  # (cumulative_cost, count, node, current_word)
        while heap:
            cost, _, node, word = heapq.heappop(heap)
            if node.is_word:
                suggestions.append(word)
            for char, child in node.children.items():
                new_cost = cost + 1 / child.freq
                heapq.heappush(heap, (new_cost, next(counter), child, word + char))
        return suggestions