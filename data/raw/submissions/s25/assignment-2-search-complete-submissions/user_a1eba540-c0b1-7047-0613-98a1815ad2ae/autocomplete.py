from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children = {}  # Dictionary to store child nodes
        self.is_word = False  # Flag to mark end of a word
        self.char_freq = {}  # To track character frequency
        self.word = ""  # To store the complete word at leaf nodes


class Autocomplete:
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.build_tree(document)
        self.suggest = self.suggest_dfs  # Default suggestion method

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()  # Create a new node if char is not present

                # Update character frequency
                if char in node.char_freq:
                    node.char_freq[char] += 1
                else:
                    node.char_freq[char] = 1
                
                node = node.children[char]  # Move to the next node
            
            node.is_word = True  # Mark the end of a word
            node.word = word  # Store the complete word at this node

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        node = self.root
        # Traverse to the prefix node
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
    
        queue = deque([(node, prefix)])  # Queue stores (node, word formed so far)
        suggestions = []

        while queue:  # Limit suggestions to 5
            current_node, word = queue.popleft()

            if current_node.is_word:
                suggestions.append(word)

            for char, child in current_node.children.items():
                 queue.append((child, word + char))

        return suggestions

    def suggest_dfs(self, prefix):
        node = self.root
        # Traverse to the prefix node
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        suggestions = []
        stack = [(node, prefix)]

        while stack:
            current_node, current_prefix = stack.pop()

            if current_node.is_word:
                suggestions.append(current_prefix)

            for char, child in current_node.children.items():
                stack.append((child, current_prefix + char))

        return suggestions

    def suggest_ucs(self, prefix):
        node = self.root
        # Traverse to the prefix node
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        pq = []
        heapq.heappush(pq, (0, prefix, node))  # (cost, word formed so far, node)
        suggestions = []

        while pq:  # Limit suggestions to 5
            cost, word, current_node = heapq.heappop(pq)  # Pop the lowest cost word

            if current_node.is_word:
                suggestions.append(word)

            total_freq = sum(current_node.char_freq.values())
            for char, child in current_node.children.items():
                freq = current_node.char_freq.get(char, 1)
                priority_cost = cost + (1 / freq)
                heapq.heappush(pq, (priority_cost, word + char, child))

        return suggestions
