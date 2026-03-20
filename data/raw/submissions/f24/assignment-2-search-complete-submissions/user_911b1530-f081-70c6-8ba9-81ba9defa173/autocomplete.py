from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children = {}
        self.frequencies = 1
        self.is_end_of_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root

            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                else:
                    node.children[char].frequencies += 1
                node = node.children[char]
            
            node.is_end_of_word = True

                

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]


    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
            
        queue = deque([node])
        queue_suffix = deque([""])
        suggestions = []

        while queue:
            current_node = queue.popleft()
            current_suffix = queue_suffix.popleft()

            if current_node.is_end_of_word:
                suggestions.append(prefix + current_suffix)
            
            for char in sorted(current_node.children.keys()):
                child_node = current_node.children[char]
                queue.append(child_node)                
                queue_suffix.append(current_suffix + char)  

        return suggestions
    

    def suggest_dfs(self, prefix):
        node = self.root
        prefix = prefix.lower()
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        
        stack_suffix = [""]
        stack = [node]
        suggestions = []

        while stack:
            current_node = stack.pop()
            current_suffix = stack_suffix.pop()

            if current_node.is_end_of_word:
                suggestions.append(prefix + current_suffix)
            
            for char in reversed(sorted(current_node.children.keys())):
                child_node = current_node.children[char]
                stack.append(child_node)                
                stack_suffix.append(current_suffix + char)  

        return suggestions


    def suggest_ucs(self, prefix):
        node = self.root
        for letter in prefix:
            if letter in node.children:
                node = node.children[letter]
            else:
                return []
        
        priority_queue = []
        heapq.heappush(priority_queue, (0, "", node))  

        suggestions = []

        while priority_queue:
            cost, current_suffix, current_node = heapq.heappop(priority_queue)

            if current_node.is_end_of_word:
                suggestions.append(prefix + current_suffix)

            for char in sorted(current_node.children.keys()):
                child_node = current_node.children[char]
                char_frequency = child_node.frequencies

                if char_frequency > 0:
                    next_cost = 1 / char_frequency  
                    heapq.heappush(priority_queue, (cost + next_cost, current_suffix + char, child_node))

        return suggestions
