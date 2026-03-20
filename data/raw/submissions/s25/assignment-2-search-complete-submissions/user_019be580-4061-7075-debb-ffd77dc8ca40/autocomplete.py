from collections import deque
from collections import defaultdict
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.c = ''
        self.weight = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    
    def build_tree(self, document):
        frequency = defaultdict(int)
        for word in document.split():
            node = self.root
            prefix = ""
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()  
                node = node.children[char]
                node.c = char
                prefix += char
                frequency[prefix] += 1            
            node.is_word = True  
        stack_node = [self.root]
        stack_char =[""]
        
        while len(stack_node) != 0:
            node = stack_node.pop()
            prefix = stack_char.pop()
            if prefix in frequency:
                node.weight = 1 / frequency[prefix] 
            for char, child in node.children.items():
                stack_node.append(child)
                stack_char.append(prefix + char)
                
            
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):        
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
        
        node_queue = [node]
        word_queue = [prefix]
        words = []
        
        while len(node_queue) != 0:
            cur_node = node_queue.pop(0)
            cur_word = word_queue.pop(0)
            
            if cur_node.is_word:
                words.append(cur_word)
            
            for char in cur_node.children:
                node_queue.append(cur_node.children[char])
                word_queue.append(cur_word +char)
            
        return words
        
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
        
        node_queue = [node]
        word_queue = [prefix]
        words = []
        
        while len(node_queue) != 0:
            cur_node = node_queue.pop()
            cur_word = word_queue.pop()
            
            if cur_node.is_word:
                words.append(cur_word)
            
            for char in cur_node.children:
                node_queue.append(cur_node.children[char])
                word_queue.append(cur_word +char)
            
        return words

            
    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
        pq = []
        heapq.heappush(pq, (0, prefix, node))
        words = []
        
        while len(pq) != 0:
            weight, word, cur_node = heapq.heappop(pq)
            
            if cur_node.is_word:
                words.append(word)
            
            for char in cur_node.children:
                heapq.heappush(pq, (weight + cur_node.children[char].weight, word + char, cur_node.children[char]))
        return words    
            
        
