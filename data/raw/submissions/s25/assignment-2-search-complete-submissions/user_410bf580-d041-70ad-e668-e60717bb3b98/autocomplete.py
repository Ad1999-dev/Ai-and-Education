from collections import deque
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
    

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs 
        #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                # update count for char
                if char in node.children:
                    child_node, count = node.children[char]
                    node.children[char] = (child_node, count + 1)
                else:
                    child_node = Node()
                    node.children[char] = (child_node, 1)
                node = child_node
            node.is_word = True  

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
    
    #Waiting for check
    def suggest_bfs(self, prefix):
        # go to the last character the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return [] # false
            node, _ = node.children[char] 

        suggest_words = []
        queue = deque()
        queue.append((node, ""))
        
        while queue and len(suggest_words) < 10:  # 10 suggestions only
            current_node, suffix = queue.popleft() # BFS, first in first out. 
            
            # Add to suggest if is_word
            if current_node.is_word:
                suggest_words.append(prefix + suffix)
            
            # Enqueue all children
            for char, (child, _) in current_node.children.items():
                queue.append((child, suffix + char))
        
        return suggest_words
    

    #Waiting for check
    def suggest_dfs(self, prefix):
        # go to the last character the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return [] # false
            node, _ = node.children[char] 
        
        suggest_words = []
        queue = deque()
        queue.append((node, ""))  # (current_node, suffix)
        
        
        while queue and len(suggest_words) < 10:  # 10 suggestions only
            current_node, suffix = queue.pop() # DFS, Last in first out. 
            
            # Add to suggest if is_word
            if current_node.is_word:
                suggest_words.append(prefix + suffix)
            
            # Push all children into stack
            for char, (child, _) in current_node.children.items():
                queue.append((child, suffix + char))
        
        return suggest_words



    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node, _ = node.children[char]  
        
        suggest_words = []
        queue = []  # priority queue: [(cumulative_cost, node, suffix)]
        queue.append((0.0, node, "")) 
        
        while queue and len(suggest_words) < 10:
            # Find the the minimum cost
            min_index = 0
            for i in range(1, len(queue)):
                if queue[i][0] < queue[min_index][0]:
                    min_index = i
            current_cost, current_node, suffix = queue.pop(min_index)

            # add to suggest_words if is_word
            if current_node.is_word:
                suggest_words.append(prefix + suffix)
            
            total_children = sum(count for _, count in current_node.children.values())
            if total_children == 0:
                continue  # Skip nodes with no children
            
            # Explore all children and append to the queue
            for char, (child_node, count) in current_node.children.items():
                edge_cost = total_children / count  # Inverse frequency
                new_cost = current_cost + edge_cost
                new_suffix = suffix + char
                queue.append((new_cost, child_node, new_suffix))
        
        return suggest_words