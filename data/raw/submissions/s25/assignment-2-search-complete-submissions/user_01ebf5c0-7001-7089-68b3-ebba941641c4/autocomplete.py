from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        word_list = document.split()
        for word in word_list:
            node = self.root
            for i, char in enumerate(word):
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                # Frequency counting
                if i + 1 < len(word):  # Check if the current character is not the last one
                    next_char = word[i + 1]
                    if next_char not in node.frequency:
                        node.frequency[next_char] = 1
                    else:
                        node.frequency[next_char] += 1
            node.is_word = True  # Mark the end of the word

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]


    def suggest_bfs(self, prefix):
        # Start from the root of the tree
        node = self.root
        # Traverse the tree to find the node associated with the last character of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]  # Move to the next node
            else:
                return []  # Return an empty list if the prefix does not exist
        suggestions = []  # List to hold the words that match the prefix
        queue = deque([(node, prefix)])  # Initialize the queue with the starting node and prefix
        # Perform BFS
        while queue:  # Continue until there are no more nodes to explore
            current_node, current_prefix = queue.popleft()  # Dequeue the front item
            # Check if the current node represents the end of a word
            if current_node.is_word:
                suggestions.append(current_prefix)  # Add the current prefix to suggestions
            # Enqueue all children of the current node
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char))  # Append child node and updated prefix to the queue
        return suggestions  # Return the list of suggestions found

    def suggest_dfs(self, prefix):
        # Start with the root of the trie
        node = self.root
        # Traverse the trie according to the characters in the prefix
        for char in prefix:
            # Check if the current character is in the children of the current node
            if char in node.children:
                # Move to the next node corresponding to the current character
                node = node.children[char]
            else:
                # If a character is not found, there are no suggestions, so return an empty list
                return []
        # List to hold all found suggestions
        suggestions = []
        # Stack for depth-first search, initialized with the current node and the prefix
        stack = [(node, prefix)]
        # Continue until there are no more nodes to explore
        while stack:
            # Pop the last node and its associated prefix from the stack
            current_node, current_prefix = stack.pop()
            # If the current node marks the end of a word, add the current prefix to suggestions
            if current_node.is_word:
                suggestions.append(current_prefix)
            # Iterate through all children nodes of the current node
            for char, child_node in current_node.children.items():
                # Push each child node along with the updated prefix onto the stack
                stack.append((child_node, current_prefix + char))
        # Return the list of suggestions found during the traversal
        return suggestions


    def suggest_ucs(self, prefix):
        # Step 1: Find the node corresponding to the last character of the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # If the prefix does not exist, return an empty list

        # Step 2: Use a priority queue to perform UCS
        suggestions = []
        priority_queue = []
        
        # Start from the found node with the prefix
        heapq.heappush(priority_queue, (0, prefix, node))  # (cost, current_prefix, current_node)

        while priority_queue:
            cost, current_prefix, current_node = heapq.heappop(priority_queue)

            # Check if this node marks the end of a word
            if current_node.is_word:
                suggestions.append(current_prefix)

            # Expand to children using the frequency for path cost
            for char, child_node in current_node.children.items():
                next_cost = cost + (1 / child_node.frequency.get(char, 1))  
                heapq.heappush(priority_queue, (next_cost, current_prefix + char, child_node))

        return suggestions
