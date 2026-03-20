from collections import deque
import heapq
import random
import string
import itertools


class Node:
    #TODO
    def __init__(self): #two attributes to contain the next character in a dictionary and to mark the end of the word
        self.children = {} 
        self.frequency = {} #to store frequencies to be used in UCS
        self.is_end_of_word = False #added extra attribute to check for the end of the word


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children: #if it is not there, then we build a new node 
                    node.children[char] = Node()

                if char not in node.frequency: #to keep track of frequencies
                    node.frequency[char] = 0
                node.frequency[char] += 1
                    
                node = node.children[char]
            node.is_end_of_word = True #mark the end of the word

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #Helper function to check if the prefix is valid
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    
    #TODO for students!!!
    def suggest_bfs(self, prefix):
        start_node = self.startsWith(prefix) #to find the start node
        if not start_node:
            return []
        suggestions = [] #initializing an empty list to keep on adding the suggestions
        queue = deque([(start_node, prefix)]) 

        while queue:
            node, current_prefix = queue.popleft()

            # If it's the end of a word, we add the current prefix to suggestions
            if node.is_end_of_word:
                suggestions.append(current_prefix)
        
            for char, child_node in node.children.items():
                queue.append((child_node, current_prefix + char))

        return suggestions
        
    

    #TODO for students!!!
    def suggest_dfs_recursive(self, prefix):
        start_node = self.startsWith(prefix)
        if not start_node:
            return []
        suggestions = []
        def dfs(node,current_prefix): #Using Recursive DFS Approach
            if node.is_end_of_word: #marks that the word is complete
                suggestions.append(current_prefix)
            for char, child_node in node.children.items():
                dfs(child_node, current_prefix + char)
        dfs(start_node, prefix)
        return suggestions

    #For trying out the stack based implementation
    
    def suggest_dfs_stack(self, prefix):
        start_node = self.startsWith(prefix)  # Finding the start node corresponding to the prefix
        if not start_node:
            return []
    
        suggestions = [] 
        stack = [(start_node, prefix)]
    
        while stack:
            node, current_prefix = stack.pop()

            if node.is_end_of_word:
                suggestions.append(current_prefix)

            for char, child_node in node.children.items():
                stack.append((child_node, current_prefix + char))
    
        return suggestions

    #TODO for students!!!

    def suggest_ucs(self, prefix):
        start_node = self.startsWith(prefix)
        if not start_node:
            return []

        suggestions = [] #empty list to append the suggestions
        pq = []
        counter = itertools.count()  #This will generate unique numbers to break ties

        heapq.heappush(pq, (0, next(counter), start_node, prefix))

        while pq:
            cost, _, node, current_prefix = heapq.heappop(pq)

            if node.is_end_of_word:
                suggestions.append(current_prefix)

            for char, child_node in node.children.items():
                if char in node.frequency:
                    char_cost = 1 / (node.frequency[char] + 1)
                else:
                    char_cost = 1

                new_cost = cost + char_cost
                heapq.heappush(pq, (new_cost, next(counter), child_node, current_prefix + char))

        return suggestions