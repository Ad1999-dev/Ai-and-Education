from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children.keys():
                    newNode  = Node()
                    node.children[char]= newNode
                node = node.children[char]
            node.is_word= True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        listOfWords = []
        node = self.root
        for letter in prefix:
            if letter not in node.children.keys():
                return listOfWords
            nextNode = node.children[letter]
            node= nextNode

        queue = deque([(node, prefix)])
        
        while len(queue) != 0:
            new_node, prefix_now = queue.popleft()
            if new_node.is_word== True:
                listOfWords.append(prefix_now)
             
            for key, value in new_node.children.items():
                queue.append((value, prefix_now + key))

        return listOfWords

            
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        listOfWords = []
        node = self.root
        for letter in prefix:
            if letter not in node.children.keys():
                return listOfWords
            nextNode = node.children[letter]
            node= nextNode
        
        stack = [(node, prefix)]

        while len(stack) != 0:
            new_node, prefix_now = stack.pop()
            if new_node.is_word== True:
                listOfWords.append(prefix_now)
             
            for key, value in new_node.children.items():
                stack.append((value, prefix_now + key))

        return listOfWords


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        listOfWords = []
        node = self.root
        for letter in prefix:
            if letter not in node.children.keys():
                return listOfWords
            nextNode = node.children[letter]
            node= nextNode
        
        cost = 0 
        priority_q = [(cost, prefix, node)]

        while len(priority_q) !=0:
            cost,word, lowestNode = heapq.heappop(priority_q)
            if lowestNode.is_word== True:
                listOfWords.append(word)

            for key, value in lowestNode.children.items():
                heapq.heappush(priority_q,(cost +1,word + key,value))

        return listOfWords