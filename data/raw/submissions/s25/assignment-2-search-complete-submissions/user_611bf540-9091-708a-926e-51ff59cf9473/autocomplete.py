from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        self.freq_map = {}

    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()    
                node = node.children[char]
                #populate the freq map, not the inverse frequency rn
                if char in self.freq_map:
                    self.freq_map[char] += 1
                else:
                    self.freq_map[char] = 1
            node.is_word = True
       

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def find_prefix(self, prefix):
        node = self.root
        running_sum = 0
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
            curr_freq = 1 / self.freq_map.get(char)
            running_sum += curr_freq
            
        return node, running_sum

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        #searches for each layer of the tree
        #when the prefix grows, the suffix gets smaller, less choices
        #traverse the tree to make up the prefix
        node, sum = self.find_prefix(prefix)
        if node is None: 
            return []
        queue = deque([(node, prefix)])
        suggestions = []
        while queue:
            node, word = queue.popleft()
            if node.is_word:
                suggestions.append(word)
            for char, child in node.children.items():
                queue.append((child, word + char))
        return suggestions
        
        
    #TODO for students!!!
    def suggest_dfs(self, prefix):
        #searches for each layer of the tree
        #when the prefix grows, the suffix gets smaller, less choices
        #traverse the tree to make up the prefix
        suggestions = []
        node, sum = self.find_prefix(prefix)
        if node is None:
            return []
        stack = [(node, prefix)]
        while stack:
            node, word = stack.pop(); 
            if node.is_word:
                suggestions.append(word)
            for char, child in node.children.items():
                stack.append((child, word + char))
             
        return suggestions
            

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        
        suggestions = []
        node, running_sum = self.find_prefix(prefix)
        if node is None: 
            return []
        pq = [(running_sum, (prefix, node))]
        while pq: 
            running_sum, (word, node) = heapq.heappop(pq)
            if node.is_word:
                suggestions.append(word)
            for char, child in node.children.items():
                curr_sum = 1 / self.freq_map.get(char, 0)
                new_cost = running_sum + curr_sum
                heapq.heappush(pq, (new_cost, (word + char, child)))
        
        return suggestions
                
            
            
            
        
