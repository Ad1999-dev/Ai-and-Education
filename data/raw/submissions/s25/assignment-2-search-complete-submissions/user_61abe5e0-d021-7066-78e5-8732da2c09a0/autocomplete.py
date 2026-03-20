from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.cost = {}

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
                    node.cost[char] = 0
        
                node.cost[char] += 1
                node = node.children[char]
            node.is_word = True
            

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def prefixNode(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        
        return node


    def suggest_bfs(self, prefix):
        node = self.prefixNode(prefix)
        queue = deque([(node, prefix)])
        suggestions = []

        if not node or not prefix:
            return []
        
        while queue:
            node, word = queue.popleft()

            if node.is_word == True:
                suggestions.append(word)
            
            for char, child in node.children.items():
                queue.append((child, word + char))
       
        return suggestions
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.prefixNode(prefix)
        suggestions = []

        if not node or not prefix:
            return []
        
        def dfs(node, word):
            if node.is_word:
                suggestions.append(word)
            
            for char, child in node.children.items():
                dfs(child, word + char) 

        dfs(node, prefix)

        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.prefixNode(prefix)
        
        if not node or not prefix:
            return []
        
        pq = []
        heapq.heappush(pq, (0, prefix, node))  
        suggestions = []
        visited = set()

        while pq:
            cost, word, current_node = heapq.heappop(pq)

            if id(current_node) in visited:
                continue
            visited.add(id(current_node))

            if current_node.is_word:
                suggestions.append(word)

            for char, child in current_node.children.items():
                new_word = word + char

                if char in current_node.cost and current_node.cost[char] > 0:
                    new_cost = cost + (1 / current_node.cost[char])
                else:
                    new_cost = cost  

                heapq.heappush(pq, (new_cost, new_word, child))

        return suggestions