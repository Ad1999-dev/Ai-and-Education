#! /usr/bin/env python3
from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.end = False
        self.cost = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        transition_freq = {}
        for word in document.split():
            node = self.root
            for char in word:
                if node not in transition_freq:
                    transition_freq[node] = {}
                if char not in transition_freq[node]:
                    transition_freq[node][char] = 0
                transition_freq[node][char] += 1
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.end = True

        for node, char_freq in transition_freq.items():
            for char, freq in char_freq.items():
                node.children[char].cost = 1 / freq
    
    
    def build_tree_cost(self, document):
        transition_freq = {}
        for word in document.split():
            node = self.root
            for char in word:
                if node not in transition_freq:
                    transition_freq[node] = {}
                if char not in transition_freq[node]:
                    transition_freq[node][char] = 0
                transition_freq[node][char] += 1
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.end = True

        for node, char_freq in transition_freq.items():
            for char, freq in char_freq.items():
                node.children[char].cost = 1 / freq

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        
        result = []
        q = deque([(node, prefix)])
        while q:
            node, word = q.popleft()
            if node.end:
                result.append(word)
            for ch, child in node.children.items():
                q.append((child, word + ch))
        return result

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        result = []
        stack = [(node, prefix)]
        while stack:
            node, word = stack.pop()
            if node.end:
                result.append(word)
            for ch, child in node.children.items():
                stack.append((child, word + ch))
        return result
        


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []  
            node = node.children[ch]

        result = []
        pq = []
        entry_count = 0  
        heapq.heappush(pq, (0, entry_count, node, prefix))

        while pq:
            cost, _, node, word = heapq.heappop(pq)  

            if node.end:
                result.append((word, cost))  
            for ch, child in node.children.items():
                new_cost = cost + child.cost
                entry_count += 1  
                heapq.heappush(pq, (new_cost, entry_count, child, word + ch))

        result.sort(key=lambda x: x[1])
        return [word for word, _ in result] 
