from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.path_cost = 0.0;   
        self.frequency = 0;

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

        if document:
            self.build_tree(document)
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node();
                node = node.children[char];
                node.frequency += 1  # Increment frequency for this character
            node.is_word = True;
        
        self.calculate_path_costs(self.root)
    
    def calculate_path_costs(self, node,  current_word=""):
        for char, child in node.children.items():
            word_to_print = current_word + char
            # Calculate path cost specifically for this child based on its frequency
            if child.frequency > 0:
                child.path_cost = (1.0 / child.frequency) + node.path_cost    # Inverse frequency normalized by siblings
            else:
                child.path_cost = node.path_cost
            print(f'Character: {char}, Word: "{word_to_print}", Frequency: {child.frequency}, Path Cost: {child.path_cost}')
            # Recur to calculate costs for children
            self.calculate_path_costs(child, word_to_print)
    

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        suggestions = []
        node = self.root

        # Traverse to the end of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions  # No suggestions, if the prefix isn't found

        # Use BFS to find all suggestions
        queue = [(node, prefix)]  # Store the node and the current word

        while queue:
            node, word = queue.pop(0)  # FIFO queue for BFS
            if node.is_word:
                suggestions.append(word)
            for char, child_node in node.children.items():
                queue.append((child_node, word + char))

        return suggestions



    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggestions = []
        node = self.root

        # Traverse to the end of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions  # No suggestions, if the prefix isn't found

        # Helper function to perform DFS recursively
        def dfs(node, word):
            if node.is_word:
                suggestions.append(word)
            for char, child_node in node.children.items():
                dfs(child_node, word + char)

        # Start the DFS
        dfs(node, prefix)
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        # Navigate to the node corresponding to the last character of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # No suggestions found

        # Use a priority queue to implement UCS
        priority_queue = []
        heapq.heappush(priority_queue, (0.0, prefix, node))  # (cost, accumulated suggestion, current node)
        suggestions = []

        while priority_queue:
            cost, current_word, current_node = heapq.heappop(priority_queue)

            # If the current node is the end of a word, add the full word to suggestions
            if current_node.is_word:
                suggestions.append(current_word)
                print(current_word, cost)
            # Traverse through children
            for char, child in current_node.children.items():
                # Push to the queue with the accumulated word
                heapq.heappush(priority_queue, (child.path_cost, current_word + char, child))
        return suggestions
            


