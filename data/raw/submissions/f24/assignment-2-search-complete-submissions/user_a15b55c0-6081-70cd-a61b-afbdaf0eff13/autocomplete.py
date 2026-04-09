from collections import deque
import heapq
import random
import string
import itertools


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.char_frequency = 0
        self.one_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.char_frequency +=1
            node.one_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        # Go to last characters node
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        bfs_queue = deque([(node, prefix)])
        suggestions = []
        while bfs_queue:
            cur_node, cur_prefix = bfs_queue.popleft()
            if cur_node.one_word:
                suggestions.append(cur_prefix)
            
            for char, child_node in cur_node.children.items():
                bfs_queue.append((child_node, cur_prefix + char))
        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        # Go to last characters node
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        dfs_stack = [(node, prefix)]
        suggestions = []
        while dfs_stack:
            cur_node, cur_prefix = dfs_stack.pop()
            if cur_node.one_word:
                suggestions.append(cur_prefix)
            
            for char, child_node in cur_node.children.items():
                dfs_stack.append((child_node, cur_prefix + char))
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        # Go to the last characters node
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        counter = itertools.count()
        suggestions = []
        priority_queue = [(0, next(counter), node, prefix)]

        # UCS Loop
        while priority_queue:
            cur_cost, count, cur_node, cur_prefix = heapq.heappop(priority_queue)

            if cur_node.one_word:
                suggestions.append(cur_prefix)
            
            # Add children nodes to the priority queue with updated costs
            for char, child_node in cur_node.children.items():
                cost = 1 / child_node.char_frequency
                heapq.heappush(priority_queue, (cur_cost + cost, next(counter), child_node, cur_prefix + char))
        
        return suggestions



