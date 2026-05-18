from collections import deque
import heapq
import itertools
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = {}

    def __lt__(self, other):
        return id(self) < id(other)  # Arbitrary but ensures stability

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for i, char in enumerate(word):
                if char not in node.children:
                    node.children[char] = Node()
                
                # Update frequency of transitions from this node
                node.frequency[char] = node.frequency.get(char, 0) + 1
                
                node = node.children[char]  # Move to the next character's node
            
            node.is_word = True

    def print_tree(self, node=None, prefix="", level=0, is_last=True, parent=None):
        if node is None:
            node = self.root

        # Generate indentation
        indent = "    " * (level - 1) + ("└── " if is_last else "├── ") if level > 0 else ""

        # Mark complete words
        marker = "*" if node.is_word else ""

        # Fetch frequency from parent's transition frequencies
        freq = ""
        if parent:  # Root has no parent
            freq_value = parent.frequency.get(prefix, 0)  # Get frequency from parent
            freq = f" ({freq_value})"

        print(f"{indent}{prefix}{marker}{freq}")

        # Get child nodes sorted alphabetically
        child_items = sorted(node.children.items())

        # Print each child node
        child_count = len(child_items)
        for i, (char, child) in enumerate(child_items):
            self.print_tree(child, prefix=char, level=level + 1, is_last=(i == child_count - 1), parent=node)




    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
    
    #TODO for students!!!
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
            cur, curPre = queue.popleft()
            if cur.is_word:
                suggestions.append(curPre)
            for char, child_node in cur.children.items():
                queue.append((child_node, curPre + char))
        return suggestions

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        suggestions = []

        def dfs(cur, path):
            if cur.is_word:
                suggestions.append(path)  # Add formed word

            for char, child in cur.children.items():
                dfs(child, path + char)  # Recur with updated path

        dfs(node, prefix)  # Start DFS from the last node of the prefix

        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        suggestions = []
        pQueue = []
        counter = itertools.count()  # Tie-breaker

        heapq.heappush(pQueue, (0, node, prefix))

        while pQueue:
            cost, cur, path = heapq.heappop(pQueue)

            if cur.is_word:
                suggestions.append(path)

            for char, child in cur.children.items():
                transition_frequency = cur.frequency.get(char, 1)  # Get frequency
                heap_cost = cost + (1 / transition_frequency)  # Lower cost for frequent transitions
                heapq.heappush(pQueue, (heap_cost, child, path + char))  # Tie-breaker

        return suggestions

