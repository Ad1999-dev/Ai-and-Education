from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.char_freq = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):

        prefix_freq = {} 
        for word in document.split():
            for i in range(len(word)):
                prefix = word[:i]
                if i < len(word):
                    if prefix not in prefix_freq:
                        prefix_freq[prefix] = {}
                    next_char = word[i]
                    prefix_freq[prefix][next_char] = prefix_freq[prefix].get(next_char, 0) + 1
        
        for word in document.split():
            node = self.root
            prefix = ""
            
            for char in word:
                prefix += char
                prev_prefix = prefix[:-1]
                
                if char not in node.children:
                    node.children[char] = Node()
                if prev_prefix in prefix_freq:
                    node.char_freq = {c: 1/freq for c, freq in prefix_freq[prev_prefix].items()}
                
                node = node.children[char]
                
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        suggestions = []
        q = deque()
        root = self.root

        for char in prefix:
            if char in root.children:
                root = root.children[char]
            else:
                return suggestions


        q.append((root, prefix))

        while q:
            node, cur_prefix = q.popleft()
            if node.is_word:
                suggestions.append(cur_prefix)
            for char, child_node in node.children.items():
                q.append((child_node, cur_prefix + char))

        return suggestions
        
    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggestions = []
        stack = []
        root = self.root

        for char in prefix:
            if char in root.children:
                root = root.children[char]
            else:
                return suggestions
        
        stack.append((root, prefix))
        while stack:
            node, cur_prefix = stack.pop()

            if node.is_word:
                suggestions.append(cur_prefix)

            for char in sorted(node.children.keys(), reverse=True):
                stack.append((node.children[char], cur_prefix + char))

        return suggestions

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        suggestions = []
        
        node = self.root
        for char in prefix:
            if char not in node.children:
                return suggestions
            node = node.children[char]
        
        pq = [(0, prefix, node)]
        heapq.heapify(pq)
        
        while pq:
            cost, word, current = heapq.heappop(pq)
            
            if current.is_word:
                suggestions.append(word)
            
            for char, child in current.children.items():
                path_cost = current.char_freq.get(char, float('inf'))
                new_cost = cost + path_cost
                heapq.heappush(pq, (new_cost, word + char, child))
        
        return suggestions