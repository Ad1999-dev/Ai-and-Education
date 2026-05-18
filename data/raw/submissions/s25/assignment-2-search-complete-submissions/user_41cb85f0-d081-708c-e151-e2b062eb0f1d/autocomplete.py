from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.cost = 0

    def __lt__(self, other): # Comparing weights for UCS return self.weight < other.weight
        return self.cost < other.cost

class Autocomplete():
    def __init__(self, parent=None, document="", char_frequency = None):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.


    def build_tree(self, document):
        for word in document.split():
            node = self. root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.cost += 1 

            node.is_word = True
  

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        #pass
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return[]

        suggestionWords = []
        queue = deque([(node, prefix)]) 

        while queue:
            current_node, current_prefix = queue.popleft()
            if current_node.is_word:
                suggestionWords.append(current_prefix)
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char)) 

        return suggestionWords

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
       #pass
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return [] 

        suggestionWords = []
        
        def dfs(current_node, current_prefix):
            if current_node.is_word:
                suggestionWords.append(current_prefix)  

            for char, child_node in current_node.children.items():
                dfs(child_node, current_prefix + char)  
        dfs(node, prefix)

        return suggestionWords


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return [] 

        suggestionWords = []
        priorityQueue = []

        heapq.heappush(priorityQueue, (node.cost, node, prefix))

        while priorityQueue:
            currentcost, current_node, current_prefix = heapq.heappop(priorityQueue)

            if current_node.is_word:
                suggestionWords.append(current_prefix)

            for char, n in current_node.children.items():
                heapq.heappush(priorityQueue, (currentcost + (float(1)/n.cost), n, current_prefix + char))

        return suggestionWords
