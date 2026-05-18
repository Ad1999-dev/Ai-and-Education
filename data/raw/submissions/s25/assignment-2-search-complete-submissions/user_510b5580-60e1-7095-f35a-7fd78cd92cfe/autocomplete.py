from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self, char = None):
        self.children = {}
        self.is_word = False
        self.char = char
   
    def __lt__(self, other):
        # Lower comparison can be by the order of the character (if required)
        return (self.char < other.char)
    
    def __gt__(self, other):
        # Lower comparison can be by the order of the character (if required)
        return (self.char > other.char)

    def __eq__(self, other):
        return (self.char == other.char)  # Simple equality check

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node(char)  # Create a new node if the character doesn't exist
                node = node.children[char]  # Move to the child node
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        
        current_node = self.root
        
        # Traverse the Trie to find the node corresponding to the last character of the prefix.
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []  # If prefix is not found, return an empty list

        # Perform BFS to find all words starting with the prefix
        words = []
        queue = [(current_node, prefix)]  # Queue of tuples (node, current_prefix)

        while queue:
            node, current_prefix = queue.pop(0)
            if node.is_word:
                words.append(current_prefix)
            for char, child_node in node.children.items():
                queue.append((child_node, current_prefix + char))

        return words
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        # Returns a list of words in Trie that start with the given prefix using DFS.
        current_node = self.root
        
        # Traverse the Trie to find the node corresponding to the last character of the prefix.
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []  # If prefix is not found, return an empty list

        # Perform DFS to find all words starting with the prefix
        words = []
        self._dfs_helper(current_node, prefix, words)
        return words

    def _dfs_helper(self, node, current_prefix, words):
        # Helper function for DFS traversal to gather words.
        if node.is_word:
            words.append(current_prefix)  # Found a complete word
        
        for char, child_node in node.children.items():
            self._dfs_helper(child_node, current_prefix + char, words)  # Recur for children


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        # Returns a list of words in Trie that start with the given prefix using Uniform Cost Search.
        current_node = self.root
        
        # Traverse the Trie to find the node corresponding to the last character of the prefix.
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []  # If prefix is not found, return an empty list

        # Use UCS to find all words starting with the prefix
        words = []
        priority_queue = [(0, current_node, prefix)]  # (cost, node, current_prefix)

        while priority_queue:
            cost, node, current_prefix = heapq.heappop(priority_queue)

            if node.is_word:
                words.append((current_prefix, cost))  # Append (word, cost)

            # Add all child nodes to the priority queue
            for char, child_node in node.children.items():
                heapq.heappush(priority_queue, (cost + 1, child_node, current_prefix + char))

        return words