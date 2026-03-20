from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.frequency = 0 
        self.path_cost = 1
        self.end = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    

    def build_tree(self, document):
        num_words = 0
        for word in document.split():
            node = self.root
            num_words += 1
            for char in word:
                lowercase = char.lower()
                # check if letter is already stored in tree
                if(lowercase not in node.children): 
                # if not, then add this new character
                    node.children[lowercase] = Node()
                # update frequency
                node.children[lowercase].frequency += 1
                # update path cost 
                node.children[lowercase].path_cost = num_words / node.children[lowercase].frequency
                # increment node to look at its next possible letters 
                node = node.children[lowercase]
            # end of word signal
            node.end = True 
            
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        curr = self.root
        frontier = [] # list of tuples (curr character, accumulated word)
        suggestions = [] 
        # find the node that stores last letter of the prefix 
        for char in prefix:
            curr = curr.children[char]
        frontier.append((curr, prefix))
        while len(frontier) > 0:
            node, word = frontier.pop(0) 
            if (node.end): # is this the end of the word?
                suggestions.append(word)
            for letter, child in node.children.items():
                frontier.append((child, word + letter)) 
        return suggestions


    #TODO for students!!!
    def suggest_dfs(self, prefix): 
        curr = self.root
        frontier = []
        suggestions = []
        # find the last letter of the prefix 
        for char in prefix:
            curr = curr.children[char]
        frontier.append((curr, prefix))
        while (len(frontier) > 0):
            node, word = frontier.pop(len(frontier) - 1) # Stack
            if (node.end):
                suggestions.append(word)
            for letter, child in node.children.items():
                frontier.append((child, word + letter))
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        curr = self.root
        heap = []
        suggestions = []
        # find the last letter of the prefix
        for char in prefix:
            curr = curr.children[char]
        heapq.heappush(heap, (curr.path_cost, curr, prefix))
        while(len(heap) > 0):
            cost, node, word = heapq.heappop(heap)
            if (node.end):
                suggestions.append(word)
            for letter, child in node.children.items():
                heapq.heappush(heap, (child.path_cost + cost, child, word + letter))
        return suggestions