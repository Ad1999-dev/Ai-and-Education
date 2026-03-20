from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.is_leaf = False
        self.frequency = 1
        self.path_cost = 0
        self.children = {}

class LeafNode(Node):
    def __init__(self):
        super().__init__()
        self.is_leaf = True

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
                else: 
                    node.children[char].frequency += 1
                node = node.children[char]
            node.children[''] = LeafNode()
        q = deque([self.root])
        while q:
            node = q.popleft()
            for child in node.children.values():
                child.path_cost = 1 / child.frequency if not child.is_leaf else 0
                q.append(child)
        

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def __find_subtree(self, prefix):
        if prefix == '':
            return None
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node
    
    def suggest_bfs(self, prefix):
        node = self.__find_subtree(prefix)
        if node is None:
            return []
        suggestions = []
        q = deque([(node, prefix)])
        while q:
            node, built_word = q.popleft()
            if node.is_leaf:
                suggestions.append(built_word)
            for char, child in node.children.items():
                q.append((child, built_word + char))
        return suggestions
    

    def suggest_dfs(self, prefix):
        def dfs_recur(node, built_word, suggestions):
            if node.is_leaf:
                suggestions.append(built_word)
            for char, child in node.children.items():
                dfs_recur(child, built_word + char, suggestions)
        node = self.__find_subtree(prefix)
        if node is None:
            return []
        suggestions = []
        dfs_recur(node, prefix, suggestions)
        return suggestions


    def suggest_ucs(self, prefix):
        node = self.__find_subtree(prefix)
        if node is None:
            return []
        suggestions = []
        pq = []
        heapq.heappush(pq, (0, prefix, 0, node))
        while pq:
            path_taken_cost, built_word, _, node = heapq.heappop(pq)
            if node.is_leaf:
                suggestions.append(built_word)
            for char, child in node.children.items():
                heapq.heappush(
                    pq, (path_taken_cost + child.path_cost, built_word + char, id(child), child)
                )
        return suggestions
