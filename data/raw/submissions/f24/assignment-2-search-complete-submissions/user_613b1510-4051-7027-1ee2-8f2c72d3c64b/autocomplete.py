from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.parent = None #letter before the current node
        self.end = False #is this node the end of a word
        self.value = '' #the letter set to the node
        self.freq = 0 #how many times the letter was repeated on this prefix
        self.children = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char in node.children:
                    #print ("found: " + char) print statements to make sure it is correctly adding and finding the chars in tree
                    node = node.children[char]
                    node.freq += 1
                    pass
                else:
                    new_node = Node()
                    new_node.parent = node
                    new_node.value = char
                    node.children[char] = new_node
                    #print ("adding: " + char)
                    node = new_node
                    node.freq += 1
                    pass
            #check if this character is the end of the word
            if char == word[len(word) - 1]:
                node.end = True
            pass

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        
        retArr = []
        node = self.root
        #finds the current node given the prefix
        if len(prefix) != 0:
            for char in prefix:
                if char in node.children:
                    node = node.children[char]
                else:
                    return []
        
        node_queue = deque([node])

        while node_queue:
            node = node_queue.popleft()
            #loops through all children and adds them to queue
            for child in node.children:
                node_queue.append(node.children[child])

            #if the node is the end of a word it travels back up the tree rebuilding the word
            if node.end:
                retword = ""
                while node != self.root:
                    retword += node.value
                    node = node.parent

                #reverses the word as it is build in reverse order
                retArr.append(retword[::-1]) 

        return retArr

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        retArr = []
        node = self.root
        #finds the current node given the prefix
        if len(prefix) != 0:
            for char in prefix:
                if char in node.children:
                    node = node.children[char]
                else:
                    return []
        
        node_queue = deque([node])

        while node_queue:
            node = node_queue.pop()
            #loops through all children and adds them to queue
            for child in node.children:
                node_queue.append(node.children[child])

            #if the node is the end of a word it travels back up the tree rebuilding the word
            if node.end:
                retword = ""
                while node != self.root:
                    retword += node.value
                    node = node.parent

                #reverses the word as it is build in reverse order
                retArr.append(retword[::-1]) 

        return retArr


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        retArr = []
        node = self.root
        #finds the current node given the prefix
        if len(prefix) != 0:
            for char in prefix:
                if char in node.children:
                    node = node.children[char]
                else:
                    return []
        
        node_queue = deque([node])

        while node_queue:
            node = max(node_queue, key = lambda obj: obj.freq)
            node_queue.remove(node)
            #loops through all children and adds them to queue
            for child in node.children:
                node_queue.append(node.children[child])

            #if the node is the end of a word it travels back up the tree rebuilding the word
            if node.end:
                retword = ""
                while node != self.root:
                    retword += node.value
                    node = node.parent

                #reverses the word as it is build in reverse order
                retArr.append(retword[::-1]) 

        return retArr


        pass
