from collections import deque
import heapq
import random
import string

# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Node:
    # Class representing a node in the trie
    def __init__(self, char=''):
        self.current_char = char  # Character stored in this node
        self.children = {}  # Dictionary to hold children nodes
        self.is_word = False  # Flag to indicate if this node marks the end of a word
        self.frequency = 0  # Frequency of the character in the trie
        self.word = ''  # The complete word represented by this node

    def __lt__(self, other):
        # Comparison method for heap operations based on frequency
        return self.frequency < other.frequency

class Autocomplete():
    # Class to handle autocomplete functionality
    def __init__(self, parent=None, document=""):
        self.root = Node()  # Initialize the root of the trie
        self.suggest = self.suggest_ucs  # Default suggestion method set to UCS
        # self.build_tree(document)  # Uncomment to build the trie from the document
    
    def build_tree(self, document):
        # Method to build the trie from a given document
        for word in document.split():  # Split document into words
            node = self.root  # Start at the root node
            for char in word:  # Iterate through each character in the word
                if char not in node.children:  # If the character is not already a child
                    node.children[char] = Node(char)  # Create a new child node
                node = node.children[char]  # Move to the child node
                node.frequency += 1  # Increment frequency of the character
            node.is_word = True  # Mark the end of the word
            node.word = word  # Store the complete word in the node

    def suggest_random(self, prefix):
        # Method to suggest random suffixes based on a prefix
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]  # Generate random suffixes
        return [prefix + suffix for suffix in random_suffixes]  # Return full suggestions with prefix
    
    #TODO for students!!!
    def suggest_bfs(self, prefix):
        # Method to suggest words using Breadth-First Search
        node = self.root  # Start at the root node
        visited = []  # List to keep track of visited words
        queue = deque([node])  # Initialize queue with the root node
        
        while queue:  # While there are nodes to process
            current_node = queue.popleft()  # Dequeue the front node

            if current_node.is_word and current_node.word.startswith(prefix):  # Check if it's a valid word
                visited.append(current_node.word)  # Add to visited list

            for _, child_node in current_node.children.items():  # Iterate through children nodes
                queue.append(child_node)  # Enqueue each child node

        return visited  # Return the list of visited words

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        # Method to suggest words using Depth-First Search
        suggestions = []  # List to store suggestions
        def dfs(node):  # Inner function for recursive DFS
            if node.is_word and node.word.startswith(prefix):  # Check if it's a valid word
                suggestions.append(node.word)  # Add to suggestions
            for _, child_node in node.children.items():  # Iterate through children nodes
                dfs(child_node)  # Recursively call DFS on child nodes

        dfs(self.root)  # Start DFS from the root
        return suggestions  # Return the list of suggestions

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        # Method to suggest words using Uniform Cost Search
        node = self.root  # Start at the root node
        for char in prefix:  # Traverse to the prefix node
            if char not in node.children:  # If character not found
                return []  # Return empty if prefix not found
            node = node.children[char]  # Move to the child node
        
        heap = []  # Initialize the priority queue
        heapq.heappush(heap, (0, node))  # Push the prefix node with cost 0
        suggestions = []  # List to store suggestions
        visited = set()  # Set to track visited nodes
        
        while heap:  # While there are nodes in the heap
            cumulative_cost, current_node = heapq.heappop(heap)  # Pop the node with the lowest cost
            
            if current_node in visited:  # If already visited, skip
                continue
            visited.add(current_node)  # Mark the current node as visited
            
            if current_node.is_word and current_node.word.startswith(prefix):  # Check if it's a valid word
                suggestions.append(current_node.word)  # Add to suggestions
            
            # Add children to heap
            for char, child in current_node.children.items():  # Iterate through children nodes
                if child not in visited and child.frequency > 0:  # Check if child is valid
                    edge_cost = 1/child.frequency  # Calculate edge cost
                    new_cost = cumulative_cost + edge_cost  # Update cumulative cost
                    heapq.heappush(heap, (new_cost, child))  # Push child to heap with new cost
        
        return suggestions  # Return the list of suggestions
    
    