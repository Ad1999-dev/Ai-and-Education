from collections import deque, defaultdict

import heapq
import random
import string




class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.freq = defaultdict(int) # frequency of children characters

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.


    def build_tree(self, document):
        for word in document.split():  # Loop through words in the document
            node = self.root
            for i, char in enumerate(word):  # Loop through each character in each word
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                if i + 1 < len(word):  # If there's a next character, increase its frequency
                    next_char = word[i + 1]
                    node.freq[next_char] += 1
            node.is_end_of_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        
        node = self.root
        for char in prefix: #loops through letters in prefix
            if char in node.children:
                node = node.children[char] # Changes current node to a child of current node
            else:
                return []  # If prefix not in trie, return empty list

        suggestions = []
        queue = deque([(node, prefix)])  # Start BFS from node that matches last char of prefix
        
        while queue:
            current_node, current_prefix = queue.popleft()
            
            if current_node.is_end_of_word:
                suggestions.append(current_prefix)
            
            for char, next_node in current_node.children.items():
                queue.append((next_node, current_prefix + char))
        print(f"Suggestions for '{prefix}': {suggestions}")
        return suggestions
            
        
        

        

    

    #TODO for students!!!
    def suggest_dfs(self, prefix): #Stack based
        node = self.root
        # Traverse to the node that matches the last character of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Return empty list if prefix not found

        suggestions = []  # Initialize the list to store suggestions
        stack = [(node, prefix)]  # Create a stack for DFS traversal, starting with the node matching the prefix

        while stack:  # Continue until the stack is empty
            current_node, current_prefix = stack.pop()  # Pop the top element from the stack

            if current_node.is_end_of_word:  # Check if we reached the end of a valid word
                suggestions.append(current_prefix)  # Add the current prefix to the suggestions list

            # Iterate through the children of the current_node
            for char, next_node in current_node.children.items():
                stack.append((next_node, current_prefix + char))  # Push the child node and updated prefix onto the stack

        print(f"Suggestions for '{prefix}': {suggestions}")  # Print the found suggestions
        return suggestions  # Return the list of suggestions

        


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        
        # Start at the root node and traverse to the node that matches the last character of the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Return empty list if the prefix is not found in the tree

        suggestions = []
        pq = []  # Priority queue for UCS
        heapq.heappush(pq, (0, prefix, node))  # Initialize with the node that matches the prefix and cost of 0

        # UCS loop, similar to BFS and DFS structure but uses a priority queue (min-heap)
        while pq:
            cost, current_prefix, current_node = heapq.heappop(pq)

            # If we reached the end of a valid word, add the current prefix to suggestions
            if current_node.is_end_of_word:
                suggestions.append(current_prefix)

            # Iterate over the children of the current node
            for char, next_node in current_node.children.items():
                # Calculate the path cost for this child node
                # The cost is the inverse of the frequency of the character following the current prefix
                if current_node.freq[char] > 0:
                    new_cost = cost + (1 / current_node.freq[char])
                else:
                    new_cost = cost + 1  # If no frequency recorded, use a default cost of 1

                # Add the child node and its cumulative cost to the priority queue
                heapq.heappush(pq, (new_cost, current_prefix + char, next_node))

        # Return the collected suggestions
        print(f"Suggestions for '{prefix}': {suggestions}")  # Print the found suggestions
        return suggestions
