from collections import deque
import heapq
import random
import string


class Node:
    # TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.edge_freq = {}  # maps char to freqeuncy count of using that edge


class Autocomplete:
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs  # Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        if document:
            self.build_tree(document)

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                if char not in node.edge_freq:
                    node.edge_freq[char] = 0

                node.edge_freq[char] += 1
                node = node.children[char]
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [
            "".join(random.choice(string.ascii_lowercase) for _ in range(3))
            for _ in range(5)
        ]
        return [prefix + suffix for suffix in random_suffixes]

    # TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        queue = deque()
        queue.append((node, prefix))

        while queue:
            current_node, current_word = queue.popleft()
            if current_node.is_word:
                suggestions.append(current_word)
            for child_char, child_node in current_node.children.items():
                queue.append((child_node, current_word + child_char))
        return suggestions

    # TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        stack = [(node, prefix)]
        while stack:
            current_node, current_word = stack.pop()
            if current_node.is_word:
                suggestions.append(current_word)
            for child_char, child_node in current_node.children.items():
                stack.append((child_node, current_word + child_char))

        return suggestions

    # TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        pq = []
        heapq.heappush(pq, (0, prefix, node))

        while pq:
            current_cost, current_word, current_node = heapq.heappop(pq)
            if current_node.is_word:
                suggestions.append(current_word)
            for child_char, child_node in current_node.children.items():
                edge_cost = 1 / current_node.edge_freq.get(child_char, 1)
                new_cost = current_cost + edge_cost
                heapq.heappush(pq, (new_cost, current_word + child_char, child_node))

        return suggestions
