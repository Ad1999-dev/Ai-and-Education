from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.isEndOfValidWord = False
        self.word = "" 
        self.frequency = 1

    def __lt__(self, other):
        return False  # arbitrary

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    def build_tree(self, document):
        for word in document.split():
            currentNode = self.root
            for letter in word:
                if letter not in list(currentNode.children.keys()):
                    currentNode.children[letter] = Node()
                currentNode = currentNode.children[letter]
                currentNode.frequency+=1
            currentNode.isEndOfValidWord = True
            currentNode.word = word
                

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        #traverse down until you hit end of prefix
        currentNode = self.root
        for letter in prefix:
            if letter not in currentNode.children:
                return [""]
            currentNode = currentNode.children[letter]

        suggestions = []
        queue = deque()

        queue.append(currentNode)
        while len(queue) > 0:
            curr = queue.popleft()
            if curr.isEndOfValidWord:
                suggestions.append(curr.word)
            for child in list(curr.children.keys()):
                queue.append(curr.children[child])
        return suggestions
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        #traverse down until you hit end of prefix
        currentNode = self.root
        for letter in prefix:
            if letter not in currentNode.children:
                return [""]
            currentNode = currentNode.children[letter]

        suggestions = []
        queue = deque()

        queue.append(currentNode)
        while len(queue) > 0:
            curr = queue.pop()
            if curr.isEndOfValidWord:
                suggestions.append(curr.word)
            for child in list(curr.children.keys()):
                queue.append(curr.children[child])
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        currentNode = self.root
        for letter in prefix:
            if letter not in currentNode.children:
                return [""]
            currentNode = currentNode.children[letter]
        suggestions = []
        heap = []
        heapq.heappush(heap,(0,currentNode)) #initial cost set to zero

        while len(heap) > 0:
            cost, curr = heapq.heappop(heap)
            if curr.isEndOfValidWord:
                suggestions.append(curr.word)
            for child in list(curr.children.keys()):
                heapq.heappush(heap,(cost + (1/curr.children[child].frequency),curr.children[child]))
        return suggestions
