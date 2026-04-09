from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children = {}
        self.isEnd = False
        self.n = 0

    def __lt__(self, other):
        return self.getCost() < other.getCost()

    def setEnd(self, isEnd):
        self.isEnd = isEnd

    def getCost(self):
        if self.n == 0:
            return float('inf')
        
        return 1 / self.n


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #Check to see if the char is in the current node's dictionary.
                #   If not, create a new node to be added to the dictionary.
                #   Either way, set the current node to the node of the current char.
                #   If we exit word loop, then we know it is the end of the word and thus that node is set to and end node
                if char is str: char = char.lower()
                else: char = str(char)
                
                if char not in node.children:
                    newNode = Node()
                    node.children[char] = newNode
                node = node.children[char]
                node.n += 1 #Increment the amount of times this char has been reached for UCS implementation
            node.setEnd(True)

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        #Navigate to the node corresponding to the last letter in the prefix
        startNode = self.navigateToStart(prefix)

        #Handle the case where the prefix's length is 0
        if startNode is None:
            return []

        queue = deque([(startNode, '')]) #FIFO Queue holding a tuple of info: (node, path so far)
        bfsSuffixes = [] #list to store the suggestion of suffixes

        while queue:
            #Choose: set curNode and curPath using queue's left-most tuple
            curNode, curPath = queue.popleft()

            #Check: if the curNode is the end of a word, add the corresponding path to the list of suffixes
            if curNode.isEnd:
                bfsSuffixes.append(curPath)

            #Explore & Enqueue: explore the child nodes and enqueue the tuple of the node and the path to that node
            for char, childNode in curNode.children.items():
                queue.append((childNode, curPath + char))

        #Finally, return the concatenation of the prefix and every suffix found
        return [prefix + suffix for suffix in bfsSuffixes]

    def suggest_dfs(self, prefix):
        #Same implementation as bfs just with a stack instead of a queue
        startNode = self.navigateToStart(prefix)

        if startNode is None:
            return []
        
        stack = deque([(startNode, '')]) #FILO Queue A.K.A a stack holding a tuple of info: (node, path so far)
        dfsSuffixes = []

        while stack:
            curNode, curPath = stack.pop() #Take right-most tuple so as to make it FILO

            if curNode.isEnd:
                dfsSuffixes.append(curPath)
            
            for char, childNode in curNode.children.items():
                stack.append((childNode, curPath + char))
            
        return [prefix + suffix for suffix in dfsSuffixes]

    def suggest_ucs(self, prefix):
        startNode = self.navigateToStart(prefix)

        if startNode is None:
            return []
        
        ucsSuffixes = []
        priorPath = 0 #Keep track of prior path cost
        minHeap = [(0, (startNode, ''))] #Use a minHeap initialized with a cost of 0 (for heap comparison) in a tuple with another tuple of the node and the path to get there
        heapq.heapify(minHeap)

        while minHeap:
            cost, (curNode, curPath) = heapq.heappop(minHeap) #Unpack the heappop 

            priorPath = cost #Update the priorPath with the current cost of the node coming off of the stack

            #Insert the path if the node is the end of a word
            if curNode.isEnd:
                ucsSuffixes.append(curPath)

            #Insert the new nodes into the minHeap making sure to update the cost as the priorPath + the cost of the node itself
            for char, childNode in curNode.children.items():
                heapq.heappush(minHeap, (priorPath + childNode.getCost(), (childNode, curPath + char)))

        return [prefix + suffix for suffix in ucsSuffixes]

    def navigateToStart(self, prefix):
        curNode = self.root

        #No prefix to work with, therefore return the root
        if (len(prefix) == 0): 
            return curNode

        #For each char in the prefix, update the curNode until you find the starting node for search
        for char in prefix:
            if char in curNode.children:
                curNode = curNode.children[char]
            else:
                return None #Prefix not found

        return curNode