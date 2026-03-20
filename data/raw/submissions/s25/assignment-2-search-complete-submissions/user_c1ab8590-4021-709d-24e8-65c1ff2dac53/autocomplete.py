from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False  # Renamed to be more descriptive.


class Autocomplete:
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs

    def build_tree(self, document):
        for word in document.split():
            clean_word = ''.join(char.lower() for char in word if char.isalpha())
            if clean_word:
                node = self.root
                for char in clean_word:
                    node = node.children.setdefault(char, Node())
                node.is_end_of_word = True

    def find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def suggest_random(self, prefix):
        return [prefix + ''.join(random.choices(string.ascii_lowercase, k=3)) for _ in range(5)]

    def suggest_bfs(self, prefix):
        node = self.find_node(prefix)
        if not node:
            return []

        suggestions = []
        queue = deque([(node, prefix)])

        while queue:
            current_node, current_prefix = queue.popleft()
            if current_node.is_end_of_word:
                suggestions.append(current_prefix)

            for char, child in current_node.children.items():
                queue.append((child, current_prefix + char))

        return suggestions

    def suggest_dfs(self, prefix):
        suggestions = []

        def dfs(node, current_prefix):
            if node.is_end_of_word:
                suggestions.append(current_prefix)
            for char, child in node.children.items():
                dfs(child, current_prefix + char)

        node = self.find_node(prefix)
        if not node:
            return []

        dfs(node, prefix)
        return suggestions

    def suggest_ucs(self, prefix):
        node = self.find_node(prefix)
        if not node:
            return []

        suggestions = []
        pq = [(len(prefix), prefix, node)]

        while pq:
            _, path, current_node = heapq.heappop(pq)

            if current_node.is_end_of_word:
                suggestions.append(path)

            for char, child in current_node.children.items():
                heapq.heappush(pq, (len(path) + 1, path + char, child))

        return suggestions