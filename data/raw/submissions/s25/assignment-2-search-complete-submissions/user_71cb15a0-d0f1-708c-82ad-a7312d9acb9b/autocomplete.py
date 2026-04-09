from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        # self.is_word = False
        self.end = False  # if the node is the end of a word -> we will set it to true 
        self.path_cost = {} # to store the path cost for each character

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        self.character_frequencies = {}  # store character frequencies in a dictionary of dictionaries

    def build_tree(self, document):
        # calculate and populate character_frequencies dict
        for word in document.split():
            for i in range(len(word) - 1):
                char = word[i]
                next_character = word[i+1]
                if char not in self.character_frequencies: 
                    self.character_frequencies[char] = {} #initialize dict in character frequencies at char
                if next_character not in self.character_frequencies[char]:
                    self.character_frequencies[char][next_character] = 0 # from char to next char is set to  0
                self.character_frequencies[char][next_character] += 1 


        for word in document.split():
            node = self.root
            for index, char in enumerate(word):
                if char not in node.children:
                    node.children[char] = Node();
                if index < len(word) - 1:
                    next_character = word[index+1]
                    if next_character not in node.children[char].path_cost:
                        # path cost = inverse of the frequency
                        if char in self.character_frequencies and next_character in self.character_frequencies[char]:
                            node.children[char].path_cost[next_character] = 1 / self.character_frequencies[char][next_character]
                        else:
                            node.children[char].path_cost[next_character] = 1  
                    
                node = node.children[char]
            node.end = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
   

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        # find the last node in prefix
        curr = self.root
        for char in prefix:
            if char in curr.children:
                curr = curr.children[char]

        if not curr:
            return []
        
        # start the BFS with a list and queue
        results = []
        queue = deque([(curr, prefix)])

        # BFS loop
        while queue:
            node, word = queue.popleft() # current node and the accumulated word
            
            #  add all completed words to our result list
            if node.end:
                results.append(word)
        
            # explore all the child nodes
            for char, child in node.children.items():
                queue.append((child, word + char)) # push children to the queue with updated word
        print(results)
        return results
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        # recursive DFS function 
        def dfs(node, word):
            if node.end:
                results.append(word) 
            
            for char, child in node.children.items():
                dfs(child, word + char)
        
        # find the last node in prefix
        curr = self.root
        for char in prefix:
            if char in curr.children:
                curr = curr.children[char]

        if not curr:
            return []
        
        results = []

        dfs(curr, prefix)
        print(results)
        return results
       


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        # find the last node in prefix
        curr = self.root
        for char in prefix:
            if char in curr.children:
                curr = curr.children[char]

        if not curr:
            return []

        results = []
        priority = [(0, prefix, curr)] 

        while priority:
            cost, curr_prefix, curr_node = heapq.heappop(priority)
            if curr_node.end: 
                results.append(curr_prefix)
            for char, child in curr_node.children.items():
                if char in curr_node.path_cost:
                    new_cost = cost + curr_node.path_cost[char]
                    heapq.heappush(priority, (new_cost, curr_prefix + char, child))
        print(results)
        return results
