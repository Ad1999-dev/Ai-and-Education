from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.char_freq = {}  # Store frequency of children characters


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                # Update frequency count
                if char in node.char_freq:
                    node.char_freq[char] += 1
                else:
                    node.char_freq[char] = 1
                node = node.children[char]
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        # Step 1: Find the node corresponding to the last character of the prefix.
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # If prefix is not found, return an empty list
        suggestions = []
        queue = deque([(node, prefix)])  # Store tuples of (current_node, current_word)

        while queue:
            current_node, current_word = queue.popleft()

            # If this node marks the end of a word, add to suggestions
            if current_node.is_word:
                suggestions.append(current_word)

            # Enqueue all children nodes
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_word + char))

        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
         # Step 1: Find the node corresponding to the last character of the prefix.
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # If prefix is not found, return an empty list
        suggestions = []
        stack = [(node, prefix)]  # Stack holds tuples of (current_node, current_prefix)

        while stack:
            current_node, current_prefix = stack.pop()

            # Check if the current node marks the end of a word
            if current_node.is_word:
                suggestions.append(current_prefix)  # Add the current word to suggestions

            # Traverse the children nodes
            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))  # Concatenate current character to prefix

        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
    # Step 1: Find the node corresponding to the last character of the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # If prefix is not found, return an empty list

    # Step 2: UCS - Use a priority queue
        pq = []
        heapq.heappush(pq, (0, prefix, node))  # (cost, word, node) -> avoids Node comparison
        suggestions = []

        while pq:
            cost, current_word, current_node = heapq.heappop(pq)

            # If this node marks the end of a word, add to suggestions
            if current_node.is_word:
                suggestions.append(current_word)

            # Add children nodes to the priority queue based on frequency
            total_freq = sum(current_node.char_freq.values())  # Get total count of all children
            for char, child_node in current_node.children.items():
                freq = current_node.char_freq.get(char, 1)
                priority_cost = cost + (1 / freq)  # Lower cost for more frequent chars
                heapq.heappush(pq, (priority_cost, current_word + char, child_node))  

        return suggestions

