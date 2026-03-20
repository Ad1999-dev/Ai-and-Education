from collections import deque
import heapq
import random
import string


class Node:
    # TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.path_cost = 0  # add a new attribute to keep track path cost of the node


class Autocomplete:
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = (
            self.suggest_ucs
        )  # Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        frequency = {}
        words = document.split()

        # Calculate frequency of each character following a prefix
        for word in words:
            for i in range(len(word)):
                prefix = word[:i]
                char = word[i]
                if prefix not in frequency:
                    frequency[prefix] = (
                        {}
                    )  # Initialize frequency dictionary for the prefix if it doesn't exist
                if char not in frequency[prefix]:
                    frequency[prefix][char] = 0
                frequency[prefix][
                    char
                ] += 1  # Increment the frequency count for the character

        # Build the tree with path cost
        for word in words:
            node = self.root
            for i in range(len(word)):
                char = word[i]
                if char not in node.children:
                    node.children[char] = (
                        Node()
                    )  # Create a new node if the character is not already a child
                node = node.children[char]  # Move to the child node
                prefix = word[:i]
                node.path_cost = (
                    1 / frequency[prefix][char]
                )  # Set the path cost based on the frequency
            node.is_word = True  # Mark  end of a word

    def suggest_random(self, prefix):
        random_suffixes = [
            "".join(random.choice(string.ascii_lowercase) for _ in range(3))
            for _ in range(5)
        ]
        return [prefix + suffix for suffix in random_suffixes]

    # TODO for students!!!
    def suggest_bfs(self, prefix):
        current = self.root
        suggestions = []

        # go to the most recent node in prefix
        for char in prefix:
            if char not in current.children:
                return []
            current = current.children[char]

        # BFS logic
        queue = deque([(current, prefix)])  # Base case: start with current node
        while queue:
            node, word = (
                queue.popleft()
            )  # Retrieve the next word and the node associated with it
            if node.is_word:
                suggestions.append(word)
            for char, child in node.children.items():
                queue.append((child, word + char))

        return suggestions

    def suggest_dfs(self, prefix):
        current = self.root
        suggestions = []
        # go to the most recent node in prefix
        for char in prefix:
            if char not in current.children:
                return []
            current = current.children[char]

        # dfs logic
        def dfs(node, word):
            if node.is_word:
                suggestions.append(word)
            for char, child in node.children.items():
                dfs(
                    child, word + char
                )  # Recursively go to next character and continue DFS

        dfs(current, prefix)  # Base case: start with current node
        return suggestions

    def suggest_ucs(self, prefix):
        current = self.root
        suggestions = []

        # go to the most recent node in prefix
        for char in prefix:
            if char not in current.children:
                return []
            current = current.children[char]

        # UCS logic
        pq = [
            (0, prefix, current)
        ]  # Base case: starts with the current node and the initial cost is 0

        while pq:
            cost, word, node = heapq.heappop(pq)
            if node.is_word:
                suggestions.append(word)

            for char, child in node.children.items():
                next_word = word + char
                heapq.heappush(pq, (cost + child.path_cost, next_word, child))

        return suggestions
