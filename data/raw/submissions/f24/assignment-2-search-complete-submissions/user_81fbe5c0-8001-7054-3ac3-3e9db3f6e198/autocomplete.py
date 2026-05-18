from collections import deque
from collections import defaultdict
import heapq
import random
import string



class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.endOfWord = False 
        self.cost = 0
        #indicate end of word, so multiple words w common prefixes dont mess each other up 


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        cost = defaultdict(int)  

        for word in document.split():
            node = self.root
            for i, char in enumerate(word):  
                if char not in node.children:
                    node.children[char] = Node()  
                
                if i < len(word) - 1:  
                    following_char = word[i + 1]
                    cost[following_char] += 1  
                    
                node = node.children[char]  
            
            node.endOfWord = True  
    #}

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):

        node = self.root

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return [] 

        queue = deque([(node, prefix)])
        ret = []

        while queue:
            
            item = queue.popleft()
            curNode = item[0]
            curPrefix = item[1]

            if curNode.endOfWord:
                ret.append(curPrefix)

            for char, childNode in curNode.children.items():
                queue.append((childNode, curPrefix + char))

        return ret

    #TODO for students!!!
    def suggest_dfs(self, prefix):

        curNode = self.root
        
        for char in prefix:
            if char in curNode.children:
                curNode = curNode.children[char]
            else:
                return []

        stack = deque([(curNode, prefix)])
        ret = []

        while stack:

            item = stack.pop()
            curNode = item[0]
            curPrefix = item[1]

            if curNode.endOfWord:
                ret.append(curPrefix)
            
            for char, childNode in curNode.children.items():
                stack.append((childNode, curPrefix + char))

        return ret


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  

        prioQueue = []
        heapq.heappush(prioQueue, (node.cost, prefix, node))  
        ret = []

        while prioQueue:
            cost, curPrefix, curNode = heapq.heappop(prioQueue)  

            if curNode.endOfWord:
                ret.append(curPrefix)

            for char, childNode in curNode.children.items():
                heapq.heappush(prioQueue, (childNode.cost, curPrefix + char, childNode))  

        return ret




                


            
