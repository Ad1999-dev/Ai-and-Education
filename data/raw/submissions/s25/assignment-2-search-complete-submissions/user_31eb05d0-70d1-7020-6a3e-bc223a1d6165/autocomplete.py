from collections import deque
import heapq
import random
import string
from collections import defaultdict
import heapq


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.path_cost = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):

        freq = defaultdict(int)
        for word in document.split():
            for i in range(1, len(word) + 1):
                freq[word[:i]] += 1


        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.path_cost = 1 / freq[word[:len(word[:i+1])]]
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]


    #TODO
    def suggest_bfs(self, prefix):
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        queue = deque([(node, prefix)])
        suggestions = []

        while queue:
            curr, word = queue.popleft()
            if curr.is_word:
                suggestions.append(word)

            for char, child in curr.children.items():
                queue.append((child, word + char))

        return suggestions
    

    #TODO
    def suggest_dfs(self, prefix):
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        stack = [(node, prefix)]
        suggestions = []

        while stack:
            curr, word = stack.pop()
            if curr.is_word:
                suggestions.append(word)
            
            for char, child in reversed(curr.children.items()):
                stack.append((child, word + char))
        
        return suggestions


    #TODO
    def suggest_ucs(self, prefix):

        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        priority_queue = [(0, node, prefix)]
        suggestions = []

        while priority_queue:
            cost, curr, word = heapq.heappop(priority_queue)
            if curr.is_word:
                suggestions.append(word)

            for char, child in curr.children.items():
                heapq.heappush(priority_queue, (cost + child.path_cost, child, word + char))

        return suggestions



