from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.freq = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    new_node = Node()
                    new_node.freq = 1
                    node.children[char] = new_node
                else:
                    node.children[char].freq += 1
                node = node.children[char]
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

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
            current, path = queue.popleft()
            if current.is_word:
                suggestions.append(path)
            for child_char, child_node in current.children.items():
                queue.append((child_node, path + child_char))
        return suggestions

    def suggest_dfs(self, prefix):

        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  
            node = node.children[char]

        suggestions = []
        def dfs(current, path):
            if current.is_word:
                suggestions.append(path)
            for char, child in current.children.items():
                dfs(child, path + char)
        
        dfs(node, prefix)
        return suggestions

    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        heap = []
        heapq.heappush(heap, (0, prefix, node))
        while heap:
            cost, path, current = heapq.heappop(heap)
            if current.is_word:
                suggestions.append(path)
            for char, child in current.children.items():
                new_cost = cost + 1 / child.freq
                heapq.heappush(heap, (new_cost, path + char, child))
        return suggestions