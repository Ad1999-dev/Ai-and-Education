from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        # self.is_word = False
        self.is_end_of_word = False
        self.cost = float('inf')  # Initialize cost to infinity or some default value
    def __lt__(self, other):
        # This compares nodes based on their cost (lowest cost should have higher priority)
        return self.cost < other.cost        

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.char_frequency = {}  # A dictionary to track character frequencies
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    # original build_tree method, works for bfs and dfs
    '''
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_end_of_word = True
'''

    # build_tree method for ucs
    def build_tree(self, document):
        words = document.split()
        
        # Step 1: Calculate frequencies of each character across all words
        for word in words:
            for char in set(word):  # Use set to count each character once per word
                if char in self.char_frequency:
                    self.char_frequency[char] += 1
                else:
                    self.char_frequency[char] = 1

        # Step 2: Assign costs as the inverse of frequencies
        char_costs = {char: 1.0 / freq for char, freq in self.char_frequency.items()}

        # Step 3: Build the trie and assign costs to nodes
        for word in words:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()  # Create a new node if it doesn't exist
                node = node.children[char]
                node.cost = char_costs[char]  # Assign the cost to the current node
            node.is_end_of_word = True  # Mark the end of the word

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
    

























    #TODO for students!!!
    def suggest_bfs(self, prefix):
        current_node = self.root
        # Step 1: Traverse to the end of the prefix
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []  # If the prefix is not found, return an empty list

        # Step 2: BFS setup
        suggestions = []
        queue = deque([(current_node, prefix)])  # Start with the (node, prefix) pair

        while queue:
            node, current_prefix = queue.popleft()  # Get the next node and the accumulated prefix

            # Step 3: If this node marks the end of a word, add it to suggestions
            if node.is_end_of_word:
                suggestions.append(current_prefix)

            # Step 4: Explore children
            for char, child_node in node.children.items():
                queue.append((child_node, current_prefix + char))  # Enqueue child nodes with updated prefix

        return suggestions

    




















    #TODO for students!!!
    def suggest_dfs(self, prefix):
        current_node = self.root
        # Step 1: Traverse to the end of the prefix
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []  # If the prefix is not found, return an empty list

        suggestions = []
        # Step 2: Start DFS from the node corresponding to the last character of the prefix
        self._dfs(current_node, prefix, suggestions)
        return suggestions

    def _dfs(self, node, current_prefix, suggestions):
        # Step 3: Check if the current node is an end of a word
        if node.is_end_of_word:
            suggestions.append(current_prefix)  # Add the current prefix to suggestions

        # Step 4: Explore each child of the current node
        for char, child_node in node.children.items():
            self._dfs(child_node, current_prefix + char, suggestions)  # Recursive call with updated prefix

















    #TODO for students!!!
    def suggest_ucs(self, prefix):
        current_node = self.root
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []

        suggestions = []
        pq = [(0, current_node, prefix)]  # (cost, node, word)
        
        while pq:
            cost, node, current_word = heapq.heappop(pq)
            if node.is_end_of_word:
                suggestions.append(current_word)
            for char, child_node in node.children.items():
                heapq.heappush(pq, (cost + child_node.cost, child_node, current_word + char))
        
        return suggestions
    














    # debugging

    def print_trie(self, node=None, prefix=''):
        if node is None:
            node = self.root  # Start from the root if no node is provided

        # If this node marks the end of a word, print the complete word
        if node.is_end_of_word:
            print(prefix)  # Print the word formed by the current prefix

        # Recursively print each child node
        for char, child_node in node.children.items():
            self.print_trie(child_node, prefix + char)



    def print_trie_structure(self, node=None, prefix='', depth=0):
        if node is None:
            node = self.root  # Start from the root if no node is provided

        # Print the current prefix and indicate if it's the end of a word
        end_marker = " (end)" if node.is_end_of_word else ""
        print("  " * depth + f"'{prefix}'{end_marker} -> [{', '.join(node.children.keys())}]")

        # Recursively print each child node
        for char, child_node in node.children.items():
            self.print_trie_structure(child_node, prefix + char, depth + 1)
