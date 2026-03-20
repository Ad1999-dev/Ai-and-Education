from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children = {}
        self.weight = 0  # Add this line to define the weight attribute
        self.is_word = False  # Add this line to define the is_word attribute
    
    def __lt__(self, other): # Comparing weights for UCS
        return self.weight < other.weight

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs  # Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:  # If character node doesn't exist
                    node.children[char] = Node()  # Create a new node for the character
                
                node = node.children[char]  # Move to the child node corresponding to the character
                
                node.weight += 1  # Increment weight for each occurrence
            
            node.is_word = True  # Mark the end of the word

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        # Start BFS from the root, navigating to the node for the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # If prefix is not found, return an empty list

        # Initialize a queue for BFS
        queue = deque([(node, prefix)])
        results = []

        while queue:
            current_node, current_prefix = queue.popleft()

            # If the current node marks the end of a word, add it to results
            if current_node.is_word:
                results.append(current_prefix)

            # Add all children to the queue for further exploration
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char))

        return results

    def suggest_dfs(self, prefix):
        # Start the DFS from the root, navigate to the node for the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # If prefix is not found, return an empty list

        # Stack to hold nodes for DFS and the corresponding prefixes
        stack = [(node, prefix)]
        results = []

        while stack:
            current_node, current_prefix = stack.pop()

            # If the current node marks the end of a word, add it to results
            if current_node.is_word:
                results.append(current_prefix)

            # Push all children to the stack for exploration
            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))

        return results
    
    def suggest_ucs(self, prefix):
        
        # Start at the root and find the node for the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # If prefix is not found, return an empty list
            node = node.children[char]
        # Priority queue to hold (cost, current_node, path)
        priority_queue = []
        heapq.heappush(priority_queue, (0, node, prefix))  # (cost, node, prefix)
        results = []

        while priority_queue:
            current_cost, current_node, current_prefix = heapq.heappop(priority_queue)

            # If we found a complete word, add it to results
            if current_node.is_word:
                results.append(current_prefix)
            # Explore all children
            for char, child_node in current_node.children.items():
                # Use the weight stored in the node
                heapq.heappush(priority_queue, (current_cost + (1/child_node.weight), child_node, current_prefix + char))
        return results