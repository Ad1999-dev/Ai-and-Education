from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.isWord = False
        self.count = 0
        self.string = ""

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    """
    Iterates through the words in the document to create a tree. For each word, iterates through the characters, adds nodes if needed, and navigates to the appropriate child to process the next letter.
    """
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            prefix = ""
            for char in word:
                prefix = prefix+char
                #TODO for students
                if char not in node.children.keys():
                    node.children[char] = Node()
                node = node.children[char]
                node.count += 1
                node.string = prefix
            node.isWord = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        # navigate to node corresponding to the end of the prefix
        node = self.root
        for char in prefix:
            if char not in node.children.keys():
                return []
            node = node.children[char]
        # bfs to traverse subtree and make a list of suggestions
        #initialize empty queue
        nodeQueue = deque()
        #add current node to the queue
        nodeQueue.append(node)
        #initialize suggestions list
        suggestions = []
        #recursively:
        while len(nodeQueue) != 0:
            node = nodeQueue.popleft()
            #add current node's children to the end of the queue
            nodeQueue.extend(node.children.values())
            #if this is the end of a word, add it to the end of the suggestions list
            if node.isWord is True:
                suggestions.append(node.string)
        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        #starts same as bfs
        node = self.root
        for char in prefix:
            if char not in node.children.keys():
                return []
            node = node.children[char]
        #rest is same except are we looking at last added node or first added
        #initialize empty queue
        nodeQueue = deque()
        #add current node to the queue
        nodeQueue.append(node)
        #initialize suggestions list
        suggestions = []
        #recursively:
        while len(nodeQueue) != 0:
            node = nodeQueue.pop()
            #add current node's children to the end of the queue
            nodeQueue.extend(node.children.values())
            #if this is the end of a word, add it to the end of the suggestions list
            if node.isWord is True:
                suggestions.append(node.string)
        return suggestions

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children.keys():
                return []
            node = node.children[char]
        heap = []
        entrycount = 0
        heapq.heappush(heap, (-1*node.count, entrycount, node))
        suggestions = []
        while len(heap) != 0:
            node = heapq.heappop(heap)[2]
            for c in node.children.values():
                heapq.heappush(heap, (-1*c.count, entrycount, c))
                entrycount += 1
            if node.isWord:
                suggestions.append(node.string)
        return suggestions

