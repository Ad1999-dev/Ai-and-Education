from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.end_of_word = False
        self.cost = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.end_of_word = True
        self.calculate_costs(self.root)

    def calculate_costs(self, node):
        
        num_children = len(node.children)
        if num_children > 0:
            node.cost = 1 / num_children  
        else:
            node.cost = 9999 

        
        for child in node.children.values():
            self.calculate_costs(child)
                
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        suggestions = []
        node = self.root
        
        for char in prefix:
            if char in node.children:
                node = node.children[char]
                
            else:
                return suggestions  #end cause no suggestions

        # Starts BFS - char by char in each word equally
        queue = deque([(node, prefix)])  

        while queue:
            current_node, current_prefix = queue.popleft()

            
            if current_node.end_of_word:
                suggestions.append(current_prefix) # adds suggestions since word is complete

            
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char))

        return suggestions


    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggestions = []
        node = self.root
        
        for char in prefix:
            if char in node.children:
                node = node.children[char]
                
            else:
                return suggestions #end cause no suggestions
            
        stack = [(node, prefix)]  
        # starts DFS using stack last in first out
        while stack:
            current_node, current_prefix = stack.pop()

            
            if current_node.end_of_word:
                suggestions.append(current_prefix) # adds suggestions since word is complete

            
            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))

        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        suggestions = []
        node = self.root
        
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions  #end cause no suggestions

        priority_queue = []
        heapq.heappush(priority_queue, (0, prefix))  # Only push the cost and prefix, exclude Node

        while priority_queue:
            total_cost, current_prefix = heapq.heappop(priority_queue)
            current_node = self.root
            for char in current_prefix:
                current_node = current_node.children[char]

            if current_node.end_of_word:
                suggestions.append(current_prefix)

            for char, child_node in current_node.children.items():
                new_cost = total_cost + child_node.cost
                heapq.heappush(priority_queue, (new_cost, current_prefix + char))

        return suggestions