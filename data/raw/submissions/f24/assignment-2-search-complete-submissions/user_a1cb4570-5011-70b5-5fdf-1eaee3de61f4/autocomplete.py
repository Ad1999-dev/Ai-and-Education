from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.freq = 0       
        self.is_word = False 


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
                node.freq += 1
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        curr_node = self.root
        for char in prefix:
            if char not in curr_node.children:
                return []
            curr_node = curr_node.children[char]
        queue = [(curr_node, prefix)]  
        suggestions = []
        counter = 0  
        while counter < len(queue):
            curr_node, word = queue[counter]
            counter += 1  
            if curr_node.is_word:
                suggestions.append(word)
            for char, child in curr_node.children.items():
                queue.append((child, word + char))
        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        curr_node = self.root
        for char in prefix:
            if char not in curr_node.children:
                return []
            curr_node = curr_node.children[char]
        suggestions = []
        def dfs(curr_node, word):
            if curr_node.is_word:
                suggestions.append(word)
            for char, child in curr_node.children.items():
                dfs(child, word + char)
        dfs(curr_node, prefix)
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        curr_node = self.root
        for char in prefix:
            if char not in curr_node.children:
                return []
            curr_node = curr_node.children[char]
        heap = []
        counter = 0  
        heapq.heappush(heap, (-curr_node.freq, counter, curr_node, prefix))
        counter += 1
        suggestions = []
        while heap:
            neg_freq, _, curr_node, word = heapq.heappop(heap)
            if curr_node.is_word:
                suggestions.append(word)
            for char, child in curr_node.children.items():
                heapq.heappush(heap, (-child.freq, counter, child, word + char))
                counter += 1
        return suggestions