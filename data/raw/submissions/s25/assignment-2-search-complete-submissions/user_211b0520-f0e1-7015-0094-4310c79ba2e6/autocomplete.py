from collections import deque
import heapq
import random
import string

class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False  # Mark end of a word
        # self.is_word = False

class Autocomplete():

    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()  # Create a new Node for the character
                node = node.children[char]
                # Each time we traverse to a child node, we increment the count of endings
                node.end_count = getattr(node, 'end_count', 0) + 1
            node.is_word = True  # Mark the last character as a complete word


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        # Find the node for the last character of the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Prefix not found

        # BFS to find all suggestions
        suggestions = []
        queue = deque([(node, prefix)])  # Start with the current node and the prefix

        while queue:
            current_node, current_prefix = queue.popleft()
            if current_node.is_word:  # If we found a complete word
                suggestions.append(current_prefix)
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char))  # Add child nodes to queue

        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        # Find the node for the last character of the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Prefix not found

        # DFS to find all suggestions
        suggestions = []

        def dfs(current_node, current_prefix):
            if current_node.is_word:  # If we found a complete word
                suggestions.append(current_prefix)
            for char, child_node in current_node.children.items():
                dfs(child_node, current_prefix + char)  # Recur for child nodes

        # Start DFS from the current node with the existing prefix
        dfs(node, prefix)
        return suggestions

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        # Find the node for the last character of the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Prefix not found

        # UCS to find all suggestions based on this prefix
        suggestions = []
        pq = []  # Priority queue (min-heap)
        
        # Using (cost, current_prefix, current_node)
        heapq.heappush(pq, (0, prefix, node))

        while pq:
            cost, current_prefix, current_node = heapq.heappop(pq)

            if current_node.is_word:  # If we found a complete word
                suggestions.append(current_prefix)
                # print(f"Suggestion: {current_prefix} | Weight (Cost): {cost}")  # Print suggestion and its cost

                # You can limit suggestions here if needed
                if len(suggestions) >= 20:  # Limit to first 20 suggestions, for example
                    return suggestions

            for char, child_node in current_node.children.items():
                # Get the number of endings from the child node to calculate the cost
                endings = getattr(child_node, 'end_count', 1)  # Default to 1 to avoid division by zero
                cost_to_child = 1 / endings  # Calculate the cost as the inverse of endings
                
                # Push items to the priority queue; avoid direct comparisons of Node instances
                heapq.heappush(pq, (cost + cost_to_child, current_prefix + char, child_node))

        return suggestions