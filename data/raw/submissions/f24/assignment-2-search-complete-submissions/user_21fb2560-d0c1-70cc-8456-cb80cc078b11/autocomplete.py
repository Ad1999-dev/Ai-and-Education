from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.parent = None 
        self.char = ''
        self.edgeCounter = 1

    def __lt__(self, node2):
        totalCost = (1/self.edgeCounter)
        totalCost2 = (1/node2.edgeCounter)
        curNode = self.parent
        while curNode != None:
            totalCost += (1/curNode.edgeCounter)
            curNode = curNode.parent
        
        node2 = node2.parent
        while node2 != None:
            totalCost2 += (1/node2.edgeCounter)
            node2 = node2.parent

        return totalCost < totalCost2

    def __gt__(self, node2):
        totalCost = (1/self.edgeCounter)
        totalCost2 = (1/node2.edgeCounter)
        curNode = self.parent
        while curNode != None:
            totalCost += (1/curNode.edgeCounter)
            curNode = curNode.parent
        
        node2 = node2.parent
        while node2 != None:
            totalCost2 += (1/node2.edgeCounter)
            node2 = node2.parent

        return totalCost > totalCost2
        



class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char in node.children:
                    node.children[char].parent = node #added line to integrate parent
                    node.edgeCounter += 1
                    node = node.children[char]
                    continue
                else:
                    node.children[char] = Node()
                    node.edgeCounter += 1
                    node.children[char].char = char
                    node.children[char].parent = node #added line to integrate parent
                    node = node.children[char]
                    continue 

            node.children['END'] = Node()
            node.edgeCounter += 1
            node.children['END'].char = "end"
            node.children['END'].parent = node


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        bfsList = []
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char] 
        #now we have reached the end of the prefix and may begin the BFS search

        cur_index = 0

        frontier = deque([])
        while cur_index != len(node.children): #for characters in h's child dictionary
            char = list(node.children)[cur_index]
            frontier.append(node.children[char]) #adds the value of each key to the frontier

            cur_index += 1

            if cur_index == len(node.children):
                while len(frontier) != 0 and frontier[0].char == "end":
                    word = ''
                    trackBack = frontier[0]
                    while trackBack != self.root:
                        trackBack = trackBack.parent
                        word += trackBack.char  
                    frontier.popleft()
                    bfsList.append(word[::-1])
    
                if len(frontier) > 0:
                    node = frontier[0]
                    cur_index = 0
                    frontier.popleft()

        return bfsList




    #TODO for students!!!
    def suggest_dfs(self, prefix):
        dfsList = []
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char] 
        #now we have reached the end of the prefix and may begin the BFS search

        cur_index = 0

        frontier = []
        while cur_index != len(node.children): #for characters in h's child dictionary
            char = list(node.children)[cur_index]
            frontier.append(node.children[char]) #adds the value of each key to the frontier

            cur_index += 1

            if cur_index == len(node.children):
                while len(frontier) != 0 and frontier[-1].char == "end":
                    word = ''
                    trackBack = frontier[-1]
                    while trackBack != self.root:
                        trackBack = trackBack.parent
                        word += trackBack.char  
                    frontier.pop()
                    dfsList.append(word[::-1])
    
                if len(frontier) > 0:
                    node = frontier[-1]
                    cur_index = 0
                    frontier.pop()

        return dfsList


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        frontier = []
        ucsList = []
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char] 
        #now we have reached the end of the prefix and may begin the BFS search

        cur_index = 0

        while cur_index != len(node.children): #for characters in h's child dictionary
            char = list(node.children)[cur_index]
            heapq.heappush(frontier, node.children[char]) #adds the value of each key to the frontier

            cur_index += 1

            if cur_index == len(node.children):
                while len(frontier) != 0 and frontier[0].char == "end":
                    word = ''
                    trackBack = frontier[0]
                    while trackBack != self.root:
                        trackBack = trackBack.parent
                        word += trackBack.char  
                    heapq.heappop(frontier)
                    ucsList.append(word[::-1])
    
                if len(frontier) > 0:
                    node = frontier[0]
                    cur_index = 0
                    heapq.heappop(frontier)

        return ucsList
