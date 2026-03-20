from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.lastChar = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                # Move to the child node
                node = node.children[char]
            # Mark the end of the word
            node.lastChar = True


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
                # If the prefix is not in the trie, return an empty list
                return []
        
        # Perform BFS to collect all words that start with the given prefix
        words = []
        queue = [(node, prefix)]  # Each entry is a tuple (node, current_word)
        
        while queue:
            current_node, current_word = queue.pop(0)

            # If the current node marks the end of a word, add the word to the list
            if current_node.lastChar:
                words.append(current_word)

            # Iterate through children of the current node
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_word + char))

        return words

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
                # Find the node corresponding to the last character of the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                # If the prefix is not in the trie, return an empty list
                return []
        
        # Perform DFS to collect all words that start with the given prefix
        words = []
        self._dfs_helper(node, prefix, words)
        return words

    def _dfs_helper(self, node, current_word, words):
        # If the current node marks the end of a word, add it to the result list
        if node.lastChar:
            words.append(current_word)

        # Traverse each child node
        for char, child_node in node.children.items():
            # Recursively build the current word and search deeper
            self._dfs_helper(child_node, current_word + char, words)


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # If the prefix is not in the trie, return an empty list

        # Perform UCS-like search using a priority queue to collect all words that start with the prefix
        words = []
        priority_queue = [(0, prefix, node)]  # (cost, current_word, current_node)

        while priority_queue:
            cost, current_word, current_node = heapq.heappop(priority_queue)

            # If the current node marks the end of a word, add it to the results
            if current_node.lastChar:
                words.append(current_word)

            # Iterate through children of the current node
            for char, child_node in current_node.children.items():
                # You could calculate the cost based on a specific metric or schemes
                next_cost = cost + ord(char)  # Example: using ASCII value as cost
                heapq.heappush(priority_queue, (next_cost, current_word + char, child_node))

        return words
