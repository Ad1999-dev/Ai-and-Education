from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self, frequency=0):
        self.children = {}
        self.frequency = frequency

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    def build_tree(self, document):
        #creating the tree
        for word in document.split():
            node = self.root
            for char in word:
                if not char in node.children:
                    node.children[char] = Node()

                node = node.children[char]
            node.children["end"] = Node()
        
        #finding the frequencies
        self.set_frequencies("", self.root)

        
    def set_frequencies(self, letter, node):
        if(letter == "end"):
            node.frequency = 1
            return 1
        else:
            count = 0
            for child in node.children:
                count = count + self.set_frequencies(child, node.children[child])
            node.frequency = count
            return count
        

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]


    def suggest_bfs(self, prefix):
        suggestions = []

        #finding prefix node
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []


        queue = deque([[prefix, node]])
        while(len(queue)>0):
            wordNode = queue.pop()
            word = wordNode[0]
            node = wordNode[1]
            for child in node.children:
                if child == "end":
                    suggestions.append(word)
                else:
                    queue.appendleft([word+child, node.children[child]])

        return suggestions

        
    def suggest_dfs(self, prefix):
        suggestions = []

        #finding prefix node
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
            

        queue = deque([[prefix, node]])
        while(len(queue)>0):
            wordNode = queue.popleft()
            word = wordNode[0]
            node = wordNode[1]
            for child in node.children:
                if child == "end":
                    suggestions.append(word)
                else:
                    queue.appendleft([word+child, node.children[child]])

        return suggestions


    def suggest_ucs(self, prefix):
        #finding prefix node
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
            
        suggestions = []
        h = [(1/node.frequency, prefix, node)]
        while(len(h) > 0):
            current = heapq.heappop(h)
            freq = current[0]
            word = current[1]
            node = current[2]
            for child in node.children:
                if child == "end":
                    suggestions.append(word)
                else:
                    heapq.heappush(h, (freq + 1/node.children[child].frequency, word+child, node.children[child]))
        
        return suggestions
