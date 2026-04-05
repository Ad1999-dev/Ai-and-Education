from collections import deque
import heapq
import random
import string
from collections import defaultdict

class Node:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False  # Track if this node marks the end of a word
        self.freq = defaultdict(int)  # Frequency of each character's occurrence

class Autocomplete():
    def __init__(self, parent=None, document="genZ.txt"):
        self.root = Node()
        self.freq = defaultdict(int)  # Frequency of characters following prefixes
        if document:
            self.build_tree(document)
        self.suggest = self.suggest_ucs # Default to BFS

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.freq[char] += 1  # Increase frequency of current char
            node.is_end_of_word = True  # Mark the end of a word
            self.freq[word] += 1  # Count the entire word for frequency

    def suggest_bfs(self, prefix):
        node = self.root
        # Traverse the tree to find the last node of the prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        queue = deque([(node, prefix)])  # Store nodes along with the word formed so far

        while queue:
            current_node, current_word = queue.popleft()
            if current_node.is_end_of_word:
                suggestions.append(current_word)
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_word + char))

        return suggestions

    def suggest_dfs(self, prefix):
        def dfs(current_node, current_word):
            if current_node.is_end_of_word:
                suggestions.append(current_word)
            for char, child_node in current_node.children.items():
                dfs(child_node, current_word + char)

        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        dfs(node, prefix)
        return suggestions

    def suggest_ucs(self, prefix):
        # Traverse the trie to find the node representing the last character of the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No suggestions if the prefix is not found
            node = node.children[char]

        suggestions = []
        priority_queue = []  # Priority queue to store nodes based on path cost

        # Initialize the priority queue with direct children of the matched prefix
        for char, child_node in node.children.items():
            cost = 1 / (child_node.freq[char] + 1)  # Inverse of frequency (lower cost for frequent characters)
            heapq.heappush(priority_queue, (cost, prefix + char, child_node))

        seen_words = set()

        while priority_queue :
            cost, current_word, current_node = heapq.heappop(priority_queue)

            if current_node.is_end_of_word and current_word not in seen_words:
                suggestions.append(current_word)
                seen_words.add(current_word)

            # Explore children of the current node
            for char, child_node in current_node.children.items():
                new_cost = cost + (1 / (child_node.freq[char] + 1))  # Accumulate path cost
                heapq.heappush(priority_queue, (new_cost, current_word + char, child_node))

        return suggestions