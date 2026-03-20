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

    # Takes a text document as input.
    # Splits the document into individual words.
    # Inserts each word into a tree (prefix tree) data structure.
    # Each character of a word becomes a node in the tree.
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

    # Implements the Breadth-First Search (BFS) algorithm on the tree.
    # Takes a prefix (the letters the user has typed so far) as input.
    # Finds all words in the tree that start with the prefix.
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        suggestions = []
        queue = deque([(node, prefix)])
        while queue:
            current_node, word = queue.popleft()  # Choose
            if current_node.is_word:            # Check
                suggestions.append(word)

            for char, child in current_node.children.items():  # Expand
                queue.append((child, word + char))

        return suggestions

    # Implements the Depth-First Search (DFS) algorithm on the tree.
    # Takes a prefix as input.
    # Finds all words in the tree that start with the prefix.
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        suggestions = []
        def dfs_recursive(current_node, current_word):
            if current_node.is_word: # Base case
                suggestions.append(current_word)
            for char, child in current_node.children.items():
                dfs_recursive(child, current_word + char)
        dfs_recursive(node, prefix)

        return suggestions

    # Implements the Uniform Cost Search (UCS) algorithm on the tree.
    # Takes a prefix as input.
    # Finds all words in the tree that start with the prefix.
    # Prioritizes suggestions based on the frequency of characters appearing after previous characters.
    def suggest_ucs(self, prefix):
        node = self.root
        cost = 0

        for char in prefix:
            if char in node.children:
                cost += 1 / node.children[char].freq
                node = node.children[char]
            else:
                return []

        suggestions = []
        counter = 0
        heap = [(cost, counter, node, prefix)]
        counter += 1
        while heap:
            cost, _, current_node, word = heapq.heappop(heap)
            if current_node.is_word:
                suggestions.append(word)

            for char, child in current_node.children.items():
                new_cost = cost + (1 / child.freq)
                heapq.heappush(heap, (new_cost, counter, child, word + char))
                counter += 1

        return suggestions
