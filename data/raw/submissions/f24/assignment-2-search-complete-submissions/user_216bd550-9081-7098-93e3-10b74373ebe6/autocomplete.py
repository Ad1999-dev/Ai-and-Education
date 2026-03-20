from collections import deque
import heapq
import random
import string

class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root  

            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                
                node = node.children[char]
                node.frequency += 1
            node.is_end_of_word = True

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
                return suggestions  
        
        queue = deque([(node, prefix)])

        while queue:
            current_node, word = queue.popleft()

            if current_node.is_end_of_word:
                suggestions.append(word)

            for char, child_node in current_node.children.items():
                queue.append((child_node, word + char))

        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggestions = []
        node = self.root

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions
        
        def dfs(node, curr):
            if node.is_end_of_word:
                suggestions.append(curr)

            for char, child_node in node.children.items():
                dfs(child_node, curr + char)

        dfs(node, prefix)

        return suggestions



    def suggest_ucs(self, prefix):
        suggestions = []
        node = self.root

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions 

        priority_q = []
        heapq.heappush(priority_q, (0, prefix, node))  

        while priority_q:
            cost, word, current_node = heapq.heappop(priority_q)

            if current_node.is_end_of_word:
                suggestions.append(word)

            for char, child_node in current_node.children.items():
                new_cost = cost + (1 / child_node.frequency)  
                heapq.heappush(priority_q, (new_cost, word + char, child_node)) 

        return suggestions