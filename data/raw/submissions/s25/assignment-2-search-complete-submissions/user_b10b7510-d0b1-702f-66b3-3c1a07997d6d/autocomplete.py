from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        #self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    #Use CEEC to check if the character is in the children of the node
    def build_tree(self, document):
        for word in document.split():
            node = self.root 
            for char in word: 
                #TODO for students
                if char not in node.children: # C: Check if the character exists in children
                    node.children[char] = Node() # E: Expand by adding a new node
                node = node.children[char]  # E: Extend to the next node
            node.children["*"] = Node()  # C: Connect by marking the end of the word

            
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
    
    #Helper Function
    def find_prefix(self,prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    #TODO for students!!!
    #Use CEEC to suggest words using BFS
    def suggest_bfs(self, prefix):
        result = []
        node = self.find_prefix(prefix)
        if not node:
            return []
        
        queue = deque()
        queue.append((node, prefix))

        while queue:
            curr_node, word = queue.popleft() 
            if '*' in curr_node.children:
                result.append(word)
            for char, child in curr_node.children.items():
                queue.append((child, word + char))
        return result

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        result = []
        node = self.find_prefix(prefix)
        if not node:
            return []
        
        stack = [(node, prefix)]

        while stack:
            curr_node, word = stack.pop() 
            if '*' in curr_node.children:
                result.append(word)
            for char, child in reversed(curr_node.children.items()): #to make sure the search starts from left to right
                stack.append((child, word + char))
        return result


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        result = []
        node = self.find_prefix(prefix)
        if not node:
            return []

        pq = []
        heapq.heappush(pq, (0, prefix, node))
        
        while pq:
            cost, word, curr_node = heapq.heappop(pq)
            if '*' in curr_node.children:
                result.append(word)
            for char, child in curr_node.children.items():
                heapq.heappush(pq, (cost + 1, word + char, child))
        return result   
