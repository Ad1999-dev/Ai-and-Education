from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children = {}
        self.parent = None #parent Node
        self.letter = None #character this node represents
        self.frequency = 0 #frequency of appearance
        # self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children: #need to add the letter
                    newNode = Node()
                    newNode.parent = node
                    newNode.letter = char
                    node.children[char] = newNode
                node = node.children[char]
                node.frequency += 1
            #word complete, add end node (if not already there)
            if "END" not in node.children:
                endNode = Node()
                endNode.parent = node
                node.children["END"] = endNode


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        #go to end-of-prefix node
        prenode = self.root
        for char in prefix:
            if char not in prenode.children:
                return [prefix]
            prenode = prenode.children[char]

        word_end_list = []
        #initialize queue
        queue = deque()
        queue.append(prenode)
        #do BFS, add all word-end nodes to list
        while queue:
            curnode = queue.popleft()
            for key in curnode.children:
                if key != "END":
                    queue.append(curnode.children[key])
                else: #has "END", so is end of a word
                    word_end_list.append(curnode)

        #create the word suggestions
        suggestions = []
        for wordend in word_end_list:
            word = ""
            curnode = wordend
            #backtrack through word
            while curnode.letter != None:
                word = curnode.letter + word
                curnode = curnode.parent
            suggestions.append(word)
        
        return suggestions
        


    def suggest_dfs(self, prefix):
        #go to end-of-prefix node
        prenode = self.root
        for char in prefix:
            if char not in prenode.children:
                return [prefix]
            prenode = prenode.children[char]

        word_end_list = []
        #initialize stack
        stack = deque()
        stack.append(prenode)
        #do DFS, add all word-end nodes to list
        while stack:
            curnode = stack.pop()
            for key in curnode.children:
                if key != "END":
                    stack.append(curnode.children[key])
                else: #has "END", so is end of a word
                    word_end_list.append(curnode)

        #create the word suggestions
        suggestions = []
        for wordend in word_end_list:
            word = ""
            curnode = wordend
            #backtrack through word
            while curnode.letter != None:
                word = curnode.letter + word
                curnode = curnode.parent
            suggestions.append(word)

        return suggestions



    def suggest_ucs(self, prefix):
        #go to end-of-prefix node
        prenode = self.root
        for char in prefix:
            if char not in prenode.children:
                return [prefix]
            prenode = prenode.children[char]

        word_end_list = []
        #initialize heap (priority queue)
        heap = []
        entrynum = 1 #just stops heapq from comparing Nodes
        heapq.heappush(heap, (0, entrynum, prenode)) #tuple is (total path cost to Node, entrynum, Node)
        #do UCS, add all word-end nodes to list
        while heap:
            curcost, _, curnode = heapq.heappop(heap)
            for key in curnode.children:
                if key != "END":
                    childnode = curnode.children[key]
                    entrynum += 1
                    heapq.heappush(heap, (curcost + (1 / childnode.frequency), entrynum, childnode))
                else: #has "END", so is end of a word
                    word_end_list.append(curnode)

        #create the word suggestions
        suggestions = []
        for wordend in word_end_list:
            word = ""
            curnode = wordend
            #backtrack through word
            while curnode.letter != None:
                word = curnode.letter + word
                curnode = curnode.parent
            suggestions.append(word)

        return suggestions
