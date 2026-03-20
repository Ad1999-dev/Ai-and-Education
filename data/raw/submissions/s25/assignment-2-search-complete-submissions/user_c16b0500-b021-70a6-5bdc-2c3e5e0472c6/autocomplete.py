from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = 0 # Init frequency to 0 (for ucs)

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        # self.suggest = self.suggest_random 
        # self.suggest = self.suggest_bfs 
        # self.suggest = self.suggest_dfs
        self.suggest = self.suggest_ucs
    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                    node.children[char].frequency = 1
                else:
                    node.children[char].frequency += 1
                node = node.children[char]
            node.is_word = True
        pass


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  #Prefix not found

        suggestions = []
        queue = deque()
        queue.append((node, prefix))
        
        while queue:
            current_node, current_word = queue.popleft()
            if current_node.is_word:
                suggestions.append(current_word)
            for char, child in current_node.children.items():
                queue.append((child, current_word + char))
        # print(suggestions) # For Report
        return suggestions

    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Prefix not found
            
        suggestions = []

        def dfs(current_node, current_word):
            if current_node.is_word:
                suggestions.append(current_word)
            for char, child in current_node.children.items():
                dfs(child, current_word + char)

        dfs(node, prefix)
        # print(suggestions) # For Report
        return suggestions

    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Prefix not found

        suggestions = []
        heap = []
        heapq.heappush(heap, (0, prefix, node))
        
        while heap:
            cost, current_word, current_node = heapq.heappop(heap)
            if current_node.is_word:
                suggestions.append(current_word)
            for char, child in current_node.children.items():
                transition_cost = 1 / child.frequency
                new_cost = cost + transition_cost
                heapq.heappush(heap, (new_cost, current_word + char, child))
        # print(suggestions) # For Report
        return suggestions
