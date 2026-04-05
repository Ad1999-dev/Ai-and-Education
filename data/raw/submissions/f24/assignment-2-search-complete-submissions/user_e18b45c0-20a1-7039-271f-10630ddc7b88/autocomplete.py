from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.char_freq = {}
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        """ Build the tree based on the document input
    
        Args:
            document (str): The document to build the tree from
        
        Returns:
            None, function modifies the tree in place, DOES NOT RETURN ANYTHING
        """
        frequencies = {}
        total_chars = 0

        # Check if document is empty
        if not document:
            print("The document is empty.")
            return

        # Calculate frequencies
        for word in document.split():
            for char in word: 
                if char in frequencies: 
                    frequencies[char] += 1
                else: 
                    frequencies[char] = 1
                total_chars += 1

        # Debug print statement to check frequencies
        print("Frequencies:", frequencies)

        # Calculate inverse frequencies
        if total_chars == 0:
            print("No characters found in the document.")
            return

        for char, freq in frequencies.items():
            self.char_freq[char] = total_chars / freq  # Store the inverse frequency

        # Debug print to ensure char_freq is populated
        print("Character Frequencies after building:", self.char_freq)

        # Build the trie
        for word in document.split():
            node = self.root
            for char in word: 
                if char not in node.children: 
                    node.children[char] = Node()  # Create new node if character is not in the children
                node = node.children[char]  # Move to next node
            node.children['*'] = None  # End of word




    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

        
        
    def suggest_bfs(self, prefix):
        """ Return a list of suggestions based on the prefix using Breadth First Search
        
        Args: 
            prefix (str): The prefix to search for
            
        Returns:
            suggestions (list): A list of suggestions based on the prefix
        """
        suggestions = [] # List to store suggestions
        node = self.root
        
        # Traverse to node of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char] # Move to next node
            else: 
                return suggestions # Return empty list if prefix is not found
            
        # BFS implementation to find suggestions
        queue = deque([(node, prefix)]) # Initialize queue with the node and prefix
        while queue: 
            current_node, current_prefix = queue.popleft() # Pop the first element from the queue
            if '*' in current_node.children:
                suggestions.append(current_prefix) # Add the prefix to the suggestions list
                
            for char, next_node in current_node.children.items():
                if char != '*': # Skip if it is the end of the word
                    queue.append((next_node, current_prefix + char)) # Add next node and prefix to the queue
                    
        return suggestions # Return the suggestions list
                
         
                
    def suggest_dfs(self, prefix):
        """ Return a list of suggestions based on the prefix using Depth First Search
        Args: 
            prefix (str): The prefix to search for
            
        Returns:
            suggestions (list): A list of suggestions based on the prefix
        """
        suggestions = [] # List to store suggestions
        node = self.root 
        
        for char in prefix: 
            if char in node.children: 
                node = node.children[char] # Move to next node
            else:
                return suggestions # Return empty list if prefix is not found
            
        # DFS to find suggestions
        stack = [(node, prefix)] # Initialize stack with the node and prefix
        while stack: 
            current_node, current_prefix = stack.pop() # Pop the last element from the stack
            if '*' in current_node.children:
                suggestions.append(current_prefix) # Add the prefix to the suggestions list
                
            for char, next_node in current_node.children.items():
                if char != '*':
                    stack.append((next_node, current_prefix + char)) # Add next node and prefix to the stack
                    
        return suggestions # Return the suggestions list
                
            

    def suggest_ucs(self, prefix):
        """ Return a list of suggestions based on the prefix using Uniform Cost Search
    
        Args: 
            prefix (str): The prefix to search for
        
        Returns:
        suggestions (list): A list of suggestions based on the prefix
        """
        suggestions = []
        node = self.root
        
        # Traverse to the node corresponding to the last character of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]  # Move to next node
            else:
                return suggestions  # Return empty list if prefix is not found
        
        # UCS implementation to find suggestions
        queue = [(0, prefix, node, prefix)]  # Modify the queue: (cost, current_prefix, node, secondary_comparator)
        
        while queue:
            cost, current_prefix, current_node, _ = heapq.heappop(queue)  # Pop elements in the new order
            if '*' in current_node.children:
                suggestions.append(current_prefix)
                    
            for char, next_node in current_node.children.items():
                if char != '*':
                    char_cost = self.char_freq.get(char, float('inf'))  # Get the cost of the character
                    # Push elements: (cost, new_prefix, next_node, secondary_comparator)
                    heapq.heappush(queue, (cost + char_cost, current_prefix + char, next_node, current_prefix + char))
        
        return suggestions  # Return the list of suggestions
