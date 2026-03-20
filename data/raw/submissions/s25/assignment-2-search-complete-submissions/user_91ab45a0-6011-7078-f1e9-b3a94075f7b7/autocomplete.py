from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        # self.value = '' #char
        self.children = {} #key is the char, value is another node
        self.is_word = False
        self.frequency = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.frequency += 1
            node.is_word = True #is the whole word
            
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
    
    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if (char not in node.children): #there are no suggestions with the given prefix
                print("No recommendations found!")
                return
            node = node.children[char]
        #at the current node who's value is the last prefix
        suggestions = [] #list of strings for output
        substrings = [] #list of substrings for output
        queue = [] #list of nodes
        for char in node.children.keys(): #gets the first level after the prefix
            queue.append(node.children[char])
            substrings.append(char)
        while (len(queue) > 0):
            node = queue.pop(0)
            if (node.is_word):
                suggestions.append(prefix + substrings[0])
            for char in node.children.keys():
                queue.append(node.children[char])
                substrings.append(substrings[0] + char)
            substrings.pop(0)
        return suggestions

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if (char not in node.children): #there are no suggestions with the given prefix
                print("No recommendations found!")
                return
            node = node.children[char]
        #at the current node who's value is the last prefix
        suggestions = []
        substrings = []
        stack = []
        for char in node.children.keys():
            stack.append(node.children[char])
            substrings.append(char)
        while (len(stack) > 0):
            node = stack.pop(len(stack) - 1)
            substr = substrings.pop(len(substrings) - 1)
            if(node.is_word):
                suggestions.append(prefix + substr)
            for char in node.children.keys():
                stack.append(node.children[char])
                substrings.append(substr + char)
        return suggestions


    #TODO for students!!!

    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if (char not in node.children): #there are no suggestions with the given prefix
                print("No recommendations found!")
                return
            node = node.children[char]
        #at the current node who's value is the last prefix
        suggestions = []
        substrings = []
        priority_queue = []
        for char in node.children.keys():
            priority_queue.append(node.children[char])
            substrings.append(char)

        def shortest_path(priority_queue): #returns the index of the node with the shortest path_cost by finding the node with the highest frequency  
            max = priority_queue[0].frequency
            maxIndex = 0
            for i in range(1, len(priority_queue)):
                if (priority_queue[i].frequency > max):
                    max = priority_queue[i].frequency
                    maxIndex = i
            return maxIndex
        
        while (len(priority_queue) > 0):
            sp = shortest_path(priority_queue)
            freq = priority_queue[sp].frequency
            node = priority_queue.pop(sp)
            substr = substrings.pop(sp)
            if(node.is_word):
                suggestions.append(prefix + substr)
                # node.is_word = False #maybe not
            for char in node.children.keys():
                priority_queue.append(node.children[char])
                priority_queue[len(priority_queue) - 1].frequency += freq
                substrings.append(substr + char)
        return suggestions

