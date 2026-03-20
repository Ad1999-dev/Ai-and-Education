from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        # self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in current_node.children:
                    current_node.children[char] = Node()
                current_node = current_node.children[char]
    #The method loops over words in the document, then over each character in each word. If a character is not in the current node’s children, it adds a new node. The tree is built with each node corresponding to a character.
    
    
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]

        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        current_node = self.root
        for char in prefix:
            if char not in current_node.children:
                return [] 
            current_node = current_node.children[char]

        suggestions = []
        queue = deque([current_node])
        
        while queue:
            node = queue.popleft()
            for char, child_node in node.children.items():
                queue.append(child_node)
                suggestions.append(prefix + char)
        return suggestions
    #We first check if the prefix exists by traversing down the tree.
    #BFS uses a queue to explore the tree level by level and gather words starting from the prefix.

    
    def dfs(node, prefix, suggestions):
        if node.is_end_of_word:
            suggestions.append(prefix)
        for char, child_node in node.children.items():
            dfs(child_node, prefix + char, suggestions)

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        current_node = self.root
        for char in prefix:
            if char not in current_node.children:
                return []  
            current_node = current_node.children[char]

        suggestions = []
        dfs(current_node, prefix, suggestions)
        return suggestions
    #The DFS is implemented recursively, where for each node, we check if it's the end of a word and then recursively call DFS on its children.
    


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        current_node = self.root
        for char in prefix:
            if char not in current_node.children:
                return []  
            current_node = current_node.children[char]

        suggestions = []
        priority_queue = []
        heapq.heappush(priority_queue, (0, current_node, prefix))

        while priority_queue:
            cost, node, current_prefix = heapq.heappop(priority_queue)
            if node.is_end_of_word:
                suggestions.append(current_prefix)
            for char, child_node in node.children.items():
                heapq.heappush(priority_queue, (cost + 1, child_node, current_prefix + char))
        return suggestions
    #UCS uses a priority queue (heap) to always expand the least costly node first. In this case, we assume a constant cost per node and use the frequency of characters to adjust the cost.
