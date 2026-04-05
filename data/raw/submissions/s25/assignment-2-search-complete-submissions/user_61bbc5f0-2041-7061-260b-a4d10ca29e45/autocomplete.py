from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {} #stores (child_node, freq)
        self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word: #iterate through chars of a word
                #TODO for students
                if char not in node.children: #whenever the sequence of chars of the word doesn't exist, we add them
                    node.children[char] = (Node(), 0)
                child_node, freq = node.children[char]
                node.children[char] = (child_node, freq + 1)
                node = node.children[char][0]
            node.is_word = True #we know it's a word after iterating all the chars of it

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return
            node = node.children[char][0]
        #after this, node will be the node containing that last letter of the prefix
        #we will run bfs on the subtree of this node
        suggestion_list = []
        queue = deque([(node, prefix)])
        while queue:
            current_node, word = queue.popleft() #remember bfs is fifo, so first in, first out
            if current_node.is_word: #if current node is a word, we add it to list
                suggestion_list.append(word)

            for char, child_node in current_node.children.items(): #get key-value pairs from the dictionary and append to queue
                queue.append((child_node[0], word+char))
        return suggestion_list
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return
            node = node.children[char][0]
        #after this, node will be the node containing that last letter of the prefix
        #we will run dfs on the subtree of this node
        suggestion_list = []
        queue = deque([(node, prefix)])
        while queue:
            current_node, word = queue.pop() #remember dfs is lifo, so last in first out
            if current_node.is_word: #if current node is a word, we add it to list
                suggestion_list.append(word)

            for char, child_node in current_node.children.items(): #get key-value pairs from the dictionary and append to queue
                queue.append((child_node[0], word+char))
        return suggestion_list


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return
            node = node.children[char][0]
        #after this, node will be the node containing that last letter of the prefix
        #we will run ucs on the subtree of this node
        suggestion_list = []
        queue = [(node, prefix, 0)] #stores (node, prefix, total_path_cost)
        while queue:
            p = float('inf')
            next_node = None
            prefix = None
            for node in queue: #choose the node with the least total path cost
                if p > node[2] or (p == node[2] and prefix > node[1]): #checks if total cost is less than current lowest and break ties using lexicographical order of word
                    p = node[2] #if it is, we update lowest total path cost, next_node and prefix
                    next_node = node
                    prefix = node[1]
            #now, we have the node corresponding to the lowest total path cost
            current_node = next_node[0]
            word = next_node[1]
            #need to delete this node from queue
            for i in range(len(queue)):
                if queue[i][1] == word:
                    queue.pop(i)
                    break
            if current_node.is_word: #if current node is a word, we add it to list
                suggestion_list.append(word)

            for char, (child_node, freq) in current_node.children.items(): #get key-value pairs from the dictionary and append to queue
                queue.append((child_node, word+char, p + 1/freq))
        return suggestion_list