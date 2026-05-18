from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.freqStore = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                    node.freqStore[char] = 1
                else:
                    node.freqStore[char] += 1
                node = node.children[char]
            node.children['!'] = Node()

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        queue = deque()
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        queue.append((node, prefix))
        suggestions = []
        while queue:
            node, prefix = queue.popleft()
            
            if '!' in node.children:
                suggestions.append(prefix)
                
            for char, child in node.children.items():
                if char != '!':
                    queue.append((child, prefix + char))
        return suggestions
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        stack = []
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        stack.append((node, prefix))
        suggestions = []
        while stack:
            node, prefix = stack.pop()
            if '!' in node.children:
                suggestions.append(prefix)
                
            for char, child in node.children.items():
                if char != '!':
                    stack.append((child, prefix + char))
                    
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        heap = []
        node = self.root
        counter = 0
    
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        heapq.heappush(heap, (0, counter, node, prefix))
        suggestions = []
        
        while heap:
            cost, _, curr_node, curr_prefix = heapq.heappop(heap)
            
            if '!' in curr_node.children:
                suggestions.append(curr_prefix)
            
            for char, child_node in curr_node.children.items():
                if char != '!':
                    freqVar = curr_node.freqStore[char]
                    new_cost = cost + (1/freqVar)
                    counter += 1
                    
                    heapq.heappush(heap, (new_cost, counter, child_node, curr_prefix + char))
    
        return suggestions