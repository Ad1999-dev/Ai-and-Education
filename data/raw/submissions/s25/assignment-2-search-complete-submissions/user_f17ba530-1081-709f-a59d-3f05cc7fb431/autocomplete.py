from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.freq = 0
        

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
                node = node.children[char]
                node.freq += 1
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        queue = deque([(node, prefix)])
        suggestions = []

        while queue:
            currentNode, currentWord = queue.popleft()
            for char, charChild in currentNode.children.items():
                newWord = currentWord + char
                queue.append((charChild, newWord))
            if currentNode.is_word:
                suggestions.append(currentWord)
        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        queue = deque([(node, prefix)])
        suggestions = []

        while queue:
            currentNode, currentWord = queue.pop()
            for char, charChild in currentNode.children.items():
                newWord = currentWord + char
                queue.append((charChild, newWord))
            if currentNode.is_word:
                suggestions.append(currentWord)
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        priorityq = []
        priorityq.append((0, node, prefix))
        suggestions = []

        while priorityq:
            priorityq.sort(key=lambda x: x[0])
            currentCost, currentNode, currentWord = priorityq.pop(0)
            if currentNode.is_word:
                suggestions.append((currentWord))
            for char, charChild in currentNode.children.items():
                pathCost = currentCost + (1 /(charChild.freq + 1))
                newWord = currentWord + char
                priorityq.append((pathCost, charChild, newWord))
        return suggestions
