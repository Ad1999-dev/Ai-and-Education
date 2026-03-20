from collections import deque
import heapq
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
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

        if document:
            self.build_tree(document)
    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_word = True 
            

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix): 
        suggestions = []
        current_node = self.root

        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return suggestions
        queue = deque([(current_node, prefix)])
        while queue:
            node, current_word = queue.popleft()
            if node.is_word: 
                suggestions.append(current_word)

            for char, child_node in node.children.items():
                queue.append((child_node, current_word + char))
        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggestions = []
        current_node = self.root

        for char in prefix: 
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return suggestions 
        
        def dfs(node, current_word):
            if node.is_word:
                suggestions.append(current_word)
            for char, child_node in node.children.items():
                dfs(child_node, current_word + char)
        dfs(current_node, prefix)
        return suggestions
    
    #TODO for students!!!
    def suggest_ucs(self, prefix):
        suggestions = []
        current_node = self.root

        for char in prefix: 
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return suggestions 

        priority_queue = [(0, prefix, id(current_node), current_node)]

        while priority_queue: 
            cost, current_word, _, node = heapq.heappop(priority_queue)

            if node.is_word: 
                suggestions.append(current_word)
            sorted_children = sorted(node.children.items())
            for child_char, child_node in sorted_children:
                new_word = current_word + child_char
                heapq.heappush(priority_queue, (cost + 1, new_word, id(child_node), child_node))
        return suggestions
        
