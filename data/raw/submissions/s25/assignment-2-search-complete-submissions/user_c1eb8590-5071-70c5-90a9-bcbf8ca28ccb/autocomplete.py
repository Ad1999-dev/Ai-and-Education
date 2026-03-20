from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if(char not in node.children):
                    node.children[char] = Node()

                if char not in node.frequency:
                    node.frequency[char] =0
                node.frequency[char] +=1
        
                node = node.children[char]               
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root #Start at root
        
        for char in prefix: #loop through each character in prefix
            if char in node.children: # if character is in the node then we move to next
                node = node.children[char]
            else:
                return [] # if the character is missing then there are no words in the prefix
        
        queue = deque([(node, prefix)]) # Use a double queue, storing a turple of node,prefix

        suggestions = [] #Store valid words for BFS 

        while queue: # loop runs until no more in queue
            curr_node,curr_pre = queue.popleft() # remove first slement from queue

            if curr_node.is_word:  # if we are at the end of a word then we add to suggestions
                suggestions.append(curr_pre)
            
            for char in curr_node.children: # iterate through each child of the current node
                queue.append((curr_node.children[char], curr_pre+char)) #+char conctaenates current word with next character 
            

        return suggestions 

    

    #TODO for students!!!
    def suggest_dfs(self, prefix): #same code as bfs but just uses a stack instead 
        node = self.root #Start at root
        
        for char in prefix: #loop through each character in prefix
            if char in node.children: # if character is in the node then we move to next
                node = node.children[char]
            else:
                return [] # if the character is missing then there are no words in the prefix
        
        stack = [(node,prefix)]
        suggestions = []

        while stack:
            curr_node,curr_pre = stack.pop() # LIFO 

            if curr_node.is_word:  # if we are at the end of a word then we add to suggestions
                suggestions.append(curr_pre)
            
            for char in curr_node.children: # iterate through each child of the current node
                stack.append((curr_node.children[char], curr_pre+char)) #+char conctaenates current word with next character 
            
        return suggestions

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root #Start at root
        
        for char in prefix: #loop through each character in prefix
            if char in node.children: # if character is in the node then we move to next
                node = node.children[char]
            else:
                return [] # if the character is missing then there are no words in the prefix
        
        queue = []
        heapq.heappush(queue, (0, prefix, node)) # put starting node to the queue
        suggestions = []

        while queue: 
            if not queue:
                return suggestions# while queue is not empty 
            cost, curr_word, curr_node = heapq.heappop(queue) # remove the lowest cost node, using the back since thats how heap works

            if curr_node.is_word:  # if we are at the end of a word then we add to suggestions
                suggestions.append(curr_word) 

            for char in curr_node.children: # go through all child of curr 
                next_node = curr_node.children[char] # move to next node 

                frequency = curr_node.frequency.get(char,1) # if char not found in frequency default to 1
                path_cost = 1/frequency # cost = 1/frequency where high frequcny have low cost and low have high costs. 

                heapq.heappush(queue, (cost + path_cost, curr_word + char, next_node)) # add child to the queue, with the new cost, the current word, and the next node  


        return suggestions
