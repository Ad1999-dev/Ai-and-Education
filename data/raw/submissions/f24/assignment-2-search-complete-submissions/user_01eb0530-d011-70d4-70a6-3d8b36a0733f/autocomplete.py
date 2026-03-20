from collections import deque, defaultdict
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.word_count = 0 #added property
        self.is_word = False #added property

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                
            node.is_word = True

        self.calculate_frequencies(self.root)
    
    def calculate_frequencies(self, node):
        if node.is_word:  
            node.word_count += 1 

        for char, child_node in node.children.items():
            self.calculate_frequencies(child_node)
            node.word_count += child_node.word_count  

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        suggest = []
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        queue = deque([(node, prefix)])

        while queue:
            curr_node, curr_prefix = queue.popleft()
            if curr_node.is_word:
                suggest.append(curr_prefix)
            for char, child_node in curr_node.children.items():
                queue.append((child_node, curr_prefix + char))

        return suggest


    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggest = []
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        self.dfs_recursive(node, prefix, suggest)

        return suggest  

    def dfs_recursive(self, node, curr_word, suggest):
        for char, child_node in node.children.items():
            self.dfs_recursive(child_node, curr_word + char, suggest)
        if node.is_word:
            suggest.append(curr_word)


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        suggest = []
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        heap = [] 
    
        heapq.heappush(heap, (0, prefix, node))
        
        while heap:
            cost, curr_prefix, curr_node = heapq.heappop(heap)
            if curr_node.is_word:
                suggest.append(curr_prefix)
            for char, child_node in curr_node.children.items():
                following_word_count = child_node.word_count
                path_cost = 1.0 / (following_word_count if following_word_count > 0 else 1) 
                new_cost = cost + path_cost
                new_prefix = curr_prefix + char
                print(new_cost, cost, new_prefix)
                heapq.heappush(heap, (new_cost, new_prefix, child_node))
        
        return suggest