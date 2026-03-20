from collections import deque
import heapq
import random
import string

class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = 0  # Initialize frequency to 0 for each new node

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            node.frequency += 1  # Root frequency increases for each word
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.frequency += 1  # Increment frequency for this path
            node.is_word = True
            # build the frequency as you go through the build tree
            # built how you are storing it, in the build tree

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        #start at the root of the trie
        node = self.root
        # Navigate down the trie following the prefix characters
        for char in prefix:
            # if character isn't found, prefix doesn't exist
            if char not in node.children:
                return [] # return empty list - no suggestions possible
            # move on to the next node
            node = node.children[char]
        # initialize empty list to store all found words
        suggestions = []
        # Create a queue starting with the node where prefix ended
        # Store both the node and the string built so far (prefix)
        queue = deque([(node, prefix)])

        # continue while there are nodes to explore
        while queue:
            # get the next node and its corresponding string from the front of the queue
            current_node, current_prefix = queue.popleft()
            # if this node marks the end of a word, add it to suggestions
            if current_node.is_word:
                suggestions.append(current_prefix)
            
            # Explore all children of current node
            # items() gives us both character and node for each child
            for char, child_node in current_node.children.items():
                # Add child to queue with its corresponding string
                queue.append((child_node, current_prefix + char))
        # Return all found suggestions
        return suggestions    
    
    def suggest_dfs(self, prefix):
        # Navigate to prefix node
        node = self.root
        # Navigate through the trie following the prefix characters
        for char in prefix:
            # If current character isn't in children, prefix doesn't exist in trie
            if char not in node.children: 
                return []  # Return empty list since no completions possible
            # Move to the next node in the path
            node = node.children[char]

        # Initialize stack with the node where prefix ends and the prefix string
        stack = [(node, prefix)]
        # Initialize empty list to store all word suggestions
        suggestions = []

        # Continue while there are nodes to explore in the stack
        while stack:
            # Get the most recently added node and its prefix from stack
            current_node, current_prefix = stack.pop()
            # If current node marks end of a word, add it to suggestions
            if current_node.is_word:
                suggestions.append(current_prefix)
            
            # Process all children of current node
            # Sort in reverse so when we pop from stack, we get lexicographic order
            for char in sorted(current_node.children.keys(), reverse=True):
                # Get the node for this character
                child_node = current_node.children[char]
                # Add to stack: (node, string_so_far + new_char)
                stack.append((child_node, current_prefix + char))
        # Return all found word completions
        return suggestions
   
    def suggest_ucs(self, prefix):
        # First, navigate to the node corresponding to the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Return empty list if prefix not found in trie
            node = node.children[char]
            
        suggestions = []
        # Priority queue stores tuples: (cost, unique_id, node, current_string)
        # Cost is inverse frequency (1/freq) so more frequent paths have lower cost
        # unique_id ensures stable sorting when costs are equal
        pq = [(1.0/node.frequency if node.frequency > 0 else float('inf'), 0, node, prefix)]
        
        # Track seen prefixes to avoid cycles and duplicates
        seen = set()
        # Counter for generating unique IDs for priority queue items
        counter = 1

        # Continue while there are paths to explore
        while pq:
            # Get the lowest cost path from the priority queue
            # cost: inverse frequency of the path
            # _: unique ID (ignored after pop)
            # current_node: current position in trie
            # current_prefix: string built up to this point
            cost, _, current_node, current_prefix = heapq.heappop(pq)
            
            # Skip if we've already seen this prefix
            if current_prefix in seen:
                continue
            seen.add(current_prefix)
            
            # If current node marks end of a word, add to suggestions
            if current_node.is_word:
                suggestions.append(current_prefix)
                
            # Explore all possible next characters in sorted order
            for char in sorted(current_node.children.keys()):
                child_node = current_node.children[char]
                new_prefix = current_prefix + char
                
                # Only explore unseen paths
                if new_prefix not in seen:
                    # Calculate new cost based on path frequency
                    # More frequent paths (higher frequency) = lower cost (1/frequency)
                    # Zero frequency paths get infinite cost
                    new_cost = 1.0/child_node.frequency if child_node.frequency > 0 else float('inf')
                    # Add new path to priority queue:
                    # - new_cost: inverse frequency of this path
                    # - counter: unique ID for stable sorting
                    # - child_node: next node in trie
                    # - new_prefix: string with new character added
                    heapq.heappush(pq, (new_cost, counter, child_node, new_prefix))
                    counter += 1
                    
        # Return all found word suggestions
        # They will be ordered by increasing path cost (decreasing frequency)
        return suggestions