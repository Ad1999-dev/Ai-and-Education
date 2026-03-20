from collections import deque
import heapq
import random
import string
import queue

class Node:
    def __init__(self, char ='', count = 1, path_cost = 0):
        self.char = char
        self.count = count
        self.children = {}
        self.is_end = False
        self.path_cost = path_cost

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node(char,1)
                else:
                    node.children[char].count += 1
                node = node.children[char]
            node.is_end = True  # Set is_end here
        
            
    
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def navigate(self, node, prefix):
        path = 0
        for char in prefix:
            if char not in node.children:
                return None
            path += 1/ node.count
            node = node.children[char]
        node.path_cost = path
        return node

    def suggest_bfs(self, prefix):
        # Step 1: Navigate to the last node of the given prefix
        start_node = self.navigate(self.root, prefix)  # Use navigate to find the correct node
        if start_node is None:
            return []  # Return an empty list if the prefix is not found
        
        suggestions = []  # Step 2: List to collect suggestions
        q = queue.Queue()  # Initialize the BFS queue
        q.put((start_node, prefix))  # Start BFS with the initial node and prefix
        
        # Step 3: BFS traversal
        while not q.empty():
            current_node, current_prefix = q.get()  # Get the next node and the prefix
            # Step 4: Check if this node marks the end of a word
            if current_node.is_end:
                suggestions.append(current_prefix)  # Add the complete prefix to suggestions
            
            # Step 5: Enqueue the children nodes
            for char, child_node in current_node.children.items():
                new_prefix = current_prefix + char  # Create new prefix
                q.put((child_node, new_prefix))  # Put the child node and new prefix in the queue

        return suggestions  # Return the collected suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        start_node = self.navigate(self.root, prefix)  # Find the node corresponding to the prefix
        if start_node is None:
            return []  # Return empty if prefix is not found
        
        suggestions = []  # List to collect suggestions
        # Inner DFS function
        def dfs(node, current_prefix):
            if node.is_end:
                suggestions.append(current_prefix)  # Add to suggestions

            for char, child_node in node.children.items():
                dfs(child_node, current_prefix + char)  # Recur for children

        # Start DFS
        dfs(start_node, prefix)

        return suggestions  # Return the suggestions

    import heapq  # Make sure to import heapq

    def suggest_ucs(self, prefix):
        start_node = self.navigate(self.root, prefix)
        suggestions = []
        
        if start_node is None:
            return suggestions 

        # Using heapq as a priority queue
        q = []
        heapq.heappush(q, (0, prefix, start_node))  # Start with path cost initialized

        while q:  # While there are items in the queue
            current_path_cost, current_prefix, current_node = heapq.heappop(q)

            # Add to suggestions if a complete word is reached
            if current_node.is_end:
                suggestions.append(current_prefix)

            # Enqueue each child while updating path costs
            for char, child_node in current_node.children.items():
                # Here you could define how you want to calculate the path cost
                child_node.path_cost = current_path_cost + (1 / child_node.count)  # Update the cost for the child node
                new_prefix = current_prefix + char
                heapq.heappush(q, (child_node.path_cost, new_prefix, child_node))

        return suggestions