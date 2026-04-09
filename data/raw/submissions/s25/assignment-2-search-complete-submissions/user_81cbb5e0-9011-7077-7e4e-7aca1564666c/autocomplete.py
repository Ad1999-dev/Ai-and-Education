from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self, char = ' '):
        self.children = {}
        self.is_word = False
        self.frequency = 1
        self.char = char

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node(char)
                node = node.children[char]
                node.frequency += 1
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        completions = []
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
            
        queue = deque([(node, prefix)])

        while queue:
            current_node, current_prefix = queue.popleft()

            if current_node.is_word:
                completions.append(current_prefix)

            for child_char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + child_char))

        return completions

    def suggest_dfs(self, prefix):
        completions = []
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
            
        stack = [(node, prefix)]

        while stack:
            current_node, current_prefix = stack.pop()

            if current_node.is_word:
                completions.append(current_prefix)

            for child_char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + child_char))

        return completions


    def suggest_ucs(self, prefix):
        completions = []
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        priority_queue = [(0, prefix, node)]

        while priority_queue:
            cost, current_prefix, current_node = heapq.heappop(priority_queue)

            if current_node.is_word:
                completions.append(current_prefix)

            for char, child in current_node.children.items():
                heapq.heappush(priority_queue, (cost + 1, current_prefix + char, child))

        return completions
