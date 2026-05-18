from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self, parent = None, letter = ""):
        self.children = {}
        self.end = False #track whether this is the end of a word
        self.parent = parent #parent node
        self.letter = letter
        self.weight = 0
    
    def __lt__(self, other):
        return self.weight < other.weight

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                node = node.children.setdefault(char, Node(parent=node, letter=char)) #traverse tree or create a new node

            #print(self.get_word(node))
            node.end = True
        
        self.assign_weights(self.root)
    
    #assign weight values for UCS
    def assign_weights(self, node):
        total = 1 if node.end else 0
        for v in node.children:
            total += self.assign_weights(node.children[v])
        node.weight = 1/total
        return total

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]


    #get the last node in the prefix
    def prefix_node(self, prefix):
        node = self.root
        for l in prefix:
            if l in node.children:
                node = node.children[l]
            else:
                return Node() #prefix doesn't exist
        return node
    
    #get word
    def get_word(self, node):
        word = ""
        cost = 0
        while node.parent != None:
            word = node.letter + word
            cost += node.weight
            node = node.parent
        print(word, cost) #print word costs
        return word
    
    #TODO for students!!!
    def suggest_bfs(self, prefix):
        start = self.prefix_node(prefix)

        queue = [start]
        suggestions = []
        while len(queue) > 0:
            node = queue.pop(0) #bfs removes first element
            if node.end == True:
                suggestions.append(self.get_word(node))
            for v in node.children:
                queue.append(node.children[v])
        
        return suggestions
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        start = self.prefix_node(prefix)

        stack = [start]
        suggestions = []
        while len(stack) > 0:
            node = stack.pop() #dfs removes last element
            if node.end == True:
                suggestions.append(self.get_word(node))
            for v in node.children:
                stack.append(node.children[v])
        
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):

        start = self.prefix_node(prefix)

        queue = [(0, start)]
        suggestions = []
        while len(queue) > 0:
            weight, node = heapq.heappop(queue)
            if node.end == True:
                suggestions.append(self.get_word(node))
            for v in node.children:
                child = node.children[v]
                heapq.heappush(queue, (weight + child.weight, child))
        
        return suggestions
