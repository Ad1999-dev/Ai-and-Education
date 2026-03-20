from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.frequency = {}
        self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        self.build_tree(document)
    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                    node.frequency[char] = 1
                else: 
                    node.frequency[char] += 1
                node = node.children[char]
            node.is_word = True

    def find_pref(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                 return None 
        return node


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.find_pref(prefix)
        que = deque([(node, prefix)])
        suggest = []
        while que:
            currNode, word = que.popleft()
            if currNode.is_word:
                suggest.append(word)
            for char, child in currNode.children.items():
                que.append((child, word + char))        
        return suggest

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.find_pref(prefix)
        stack = [(node, prefix)]
        suggest = []
        while stack:
            currNode, word = stack.pop()
            if currNode.is_word:
                suggest.append(word)
            for char, child in reversed(currNode.children.items()):
                stack.append((child, word + char))   
        return suggest


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.find_pref(prefix)
        if not node:
            return []  
        prique = [(0, 0, node, prefix)] 
        suggest = []
        ix = 1
        while prique:
            cost, _, curr, word = heapq.heappop(prique)
            if curr.is_word:
                suggest.append(word)
            for char, child in curr.children.items():
                freq = curr.frequency.get(char, 1) 
                heapq.heappush(prique, (cost + (1 / freq), ix, child, word + char))
                ix += 1
        return suggest
