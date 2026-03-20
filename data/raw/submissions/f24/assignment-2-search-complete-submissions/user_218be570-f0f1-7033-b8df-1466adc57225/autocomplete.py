from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        #added attributes
        #keeps track of whether a node marks the end of a complete word
        self.isWord = False
        #frequency of each character
        self.frequency = 0
        
class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.


    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                #convert everything to lowercase to handle searches in any case
                char = char.lower()
                #if a character hasn't already been added, add it
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                #increment frequency of the node
                node.frequency += 1
            #mark the end of a complete word
            node.isWord = True


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix+suffix for suffix in random_suffixes]


    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        #check if there are any words in the text with the given prefix  
        #move down the tree to start at the end of the prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        #queue of the tuple tree node and the word built so far
        queue = deque([(node, prefix)])
        #array of suggestions
        suggestions = []
        
        while queue:
            #get the first word from the queue
            currNode, currWord = queue.popleft()

            #add the word to suggestions if it is a complete word from the text
            if currNode.isWord:
                suggestions.append(currWord)

            #add all the children of the current node to the queue
            #append the child character to the current word to form the new word
            for char, node in currNode.children.items():
                queue.append((node, currWord+char))

        return suggestions

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        #check if there are any words in the text with the given prefix  
        #move down the tree to start at the end of the prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        #array of suggestions
        suggestions = []

        #helper recursive dfs function
        def dfs(currNode, currWord):
            #add the word to suggestions if it is a complete word from the text
            if currNode.isWord:
                suggestions.append(currWord)

            #explore each child node
            for char, node in currNode.children.items():
                dfs(node, currWord+char)

        #begin dfs from the node of the end of prefix
        dfs(node, prefix)
        return suggestions

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        #check if there are any words in the text with the given prefix  
        #move down the tree to start at the end of the prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        #minimum priority queue containing the tuple cost, tree node, and word built so far
        priorityQueue = []
        heapq.heappush(priorityQueue, (0, prefix, node))
        suggestions = []

        while priorityQueue:
            #get the node with the least cost
            cost, currWord, currNode = heapq.heappop(priorityQueue)

            #add the word to suggestions if it is a complete word from the text
            if currNode.isWord:
                suggestions.append(currWord)

            #add all the children to the priority queue along with their cost
            for char, node in currNode.children.items():
                #cost = 1/frequency
                newCost = cost + 1 / node.frequency
                heapq.heappush(priorityQueue, (newCost, currWord+char, node))

        return suggestions