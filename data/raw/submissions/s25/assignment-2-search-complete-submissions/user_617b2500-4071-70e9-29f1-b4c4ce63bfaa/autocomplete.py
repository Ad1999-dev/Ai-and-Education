from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:    
                    node.children[char] = Node()
                if char in node.frequency:
                    node.frequency[char] =  node.frequency[char] + 1
                else:
                    node.frequency[char] = 1
                node = node.children[char]
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        rec =[]
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else: 
                return rec
        queue = deque([(node, prefix)])

        while queue:
            current_node, word = queue.popleft()
            if current_node.is_word:
                rec.append(word)
            for char, child_node in current_node.children.items():
                queue.append((child_node, word + char))
        return rec
        
        
    #TODO for students!!!
    def suggest_dfs(self, prefix):
        rec = []  
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return rec  
        stack = [(node, prefix)]  

        while stack:
            current_node, word = stack.pop() 
            if current_node.is_word:
                rec.append(word)
            for char, child_node in reversed(current_node.children.items()):
                stack.append((child_node, word + char))
        return rec

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        rec = []
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return rec 
    
        heap = []
        heapq.heappush(heap, (0, prefix, node))  

        while heap:
            cost, currentWord, currentNode = heapq.heappop(heap)  

            if currentNode.is_word:
                rec.append(currentWord)
        
            for char, childNode in currentNode.children.items():
                newCost = cost + (1 / currentNode.frequency[char])  
                heapq.heappush(heap, (newCost, currentWord + char, childNode))  

        return rec