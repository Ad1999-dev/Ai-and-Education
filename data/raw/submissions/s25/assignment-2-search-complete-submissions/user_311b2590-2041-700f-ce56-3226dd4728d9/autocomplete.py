from collections import deque
import heapq
import random
import string

class PQNode:
    def __init__(self, inv_freq, node, word):
        self.inv_freq = inv_freq
        self.node = node
        self.word = word
        
    def __lt__(self, other):
        return self.inv_freq <= other.inv_freq
    


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.freq = 0
        self.inv_freq = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if(char not in node.children):
                    node.children[char] = Node()
                node = node.children[char]
                node.freq += 1
                node.inv_freq = 1/node.freq
                if(word[-1] == char):
                    node.is_word = True
                

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            node = node.children[char]
        
        node_queue = deque([node])
        word_queue = deque([prefix])
        suggestions = []
        
        while(node_queue):
            cur_node = node_queue.popleft()
            cur_word = word_queue.popleft()
            for child_key, child_val in cur_node.children.items():
                node_queue.append(child_val)
                word_queue.append(cur_word + child_key)
                if(child_val.is_word):
                    suggestions.append(cur_word + child_key)
        return suggestions
                
                
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            node = node.children[char]
            
        node_queue = deque([node])
        word_queue = deque([prefix])
        suggestions = []
        
        while(node_queue):
            cur_node = node_queue.pop()
            cur_word = word_queue.pop()
            for child_key, child_val in cur_node.children.items():
                node_queue.append(child_val)
                word_queue.append(cur_word + child_key)
                if(child_val.is_word):
                    suggestions.append(cur_word + child_key)
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            node = node.children[char]
            
        node_pq = [PQNode(node.inv_freq, node, prefix)]
        heapq.heapify(node_pq)
        suggestions = []
        
        while(node_pq):
            cur = heapq.heappop(node_pq)
            cur_node = cur.node
            cur_word = cur.word
            for child_key, child_val in cur_node.children.items():
                heapq.heappush(node_pq, PQNode(child_val.inv_freq, child_val, cur_word + child_key))
                if(child_val.is_word):
                    suggestions.append(cur_word + child_key)
        return suggestions
            
            
            
            
        
            
        


