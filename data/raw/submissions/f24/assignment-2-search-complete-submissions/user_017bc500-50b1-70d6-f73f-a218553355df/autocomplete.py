from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.end = False
        self.prio = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char in node.children:
                    node = node.children[char]
                else:
                    node.children[char] = Node()
                    node = node.children[char]
                node.prio = node.prio + 1
            node.end = True


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        curr = self.root
        #navigate to current location in word
        for char in prefix:
            curr = curr.children[char]
        queue = []
        for key in curr.children.keys():
            queue.append([prefix + key, curr.children[key]])
        if(len(queue) == 0):
            return []
        completed = []
        while len(queue) != 0:
            currWord = queue.pop(0)
            if currWord[1].end == True:
                completed.append(currWord[0])
            if currWord[1].children:
                for key in currWord[1].children.keys():
                    queue.append([currWord[0] + key, currWord[1].children[key]])
        return completed

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        curr = self.root
        #navigate to current location in word
        for char in prefix:
            curr = curr.children[char]
        stack = []
        for key in curr.children.keys():
            stack.append([prefix + key, curr.children[key]])
        if(len(stack) == 0):
            return []
        completed = []
        while len(stack) != 0:
            currWord = stack.pop()
            if currWord[1].end == True:
                completed.append(currWord[0])
            if currWord[1].children:
                for key in currWord[1].children.keys():
                    stack.append([currWord[0] + key, currWord[1].children[key]])
        return completed



    #TODO for students!!!
    def suggest_ucs(self, prefix):
        curr = self.root
        for char in prefix:
            curr = curr.children[char]

        priority_queue = []
        
        for key in curr.children.keys():
            child = curr.children[key]
            heapq.heappush(priority_queue, (1/child.prio, prefix + key, child))
        
        completed = []

        while priority_queue:
            prio, currWord, currNode = heapq.heappop(priority_queue)

            if currNode.end == True:
                completed.append(currWord)
            for key in currNode.children.keys():
                heapq.heappush(priority_queue,(prio + 1/currNode.children[key].prio, currWord + key, currNode.children[key]))
            
        return completed




        

