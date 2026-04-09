from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        # self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document="genZ.txt"):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                    node.freqs[char] = 0

                    node.freqs[char] += 1
                    node = node.children[char]

            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return[]
            node = node.children[char]

        suggestions = []
        queue = deque()
        queue.append((node, prefix))

        while queue:
            current, word = queue.popleft()
            if current.is_word:
                suggestions.append(word)
            for char, child in current.children.items():
                queue.append((child, word + char))
        return suggestions


    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  
            node = node.children[char]

        suggestions = []
        def dfs(current_node, current_word):
            if current_node.is_word:
                suggestions.append(current_word)
            for char, child in current_node.children.items():
                dfs(child, current_word + char)
        dfs(node, prefix)
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  
            node = node.children[char]

        suggestions = []
        heap = []
        heapq.heappush(heap, (0, node, prefix))

        while heap:
            cost, current, word = heapq.heappop(heap)
            if current.is_word:
                suggestions.append(word)
            for char, child in current.children.items():
                freq = current.freqs[char]
                edge_cost = 1 / freq  
                heapq.heappush(heap, (cost + edge_cost, child, word + char))
        return suggestions
